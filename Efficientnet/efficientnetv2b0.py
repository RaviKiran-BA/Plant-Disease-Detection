import os
import json
import random
import shutil
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as effv2_preprocess
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
)

# -----------------------------
# Determinism and CPU-only
# -----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Force CPU only
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
# Optional: stricter reproducibility, may slow down CPU
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# -----------------------------
# Paths and config
# -----------------------------
DATASET_DIR = r"C:\Users\DELL\Downloads\New Plant Diseases Dataset(Augmented)"
HOME = os.path.expanduser("~")
OUTPUT_DIR = os.path.join(HOME, "plant_disease_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# EfficientNetV2B0 outputs
EFFICIENTNETV2_DIR = os.path.join(OUTPUT_DIR, "efficientnetv2b0")
os.makedirs(EFFICIENTNETV2_DIR, exist_ok=True)
EFFICIENTNETV2_MODEL_HEAD = os.path.join(EFFICIENTNETV2_DIR, "plant_disease_efficientnetv2b0_head.keras")
EFFICIENTNETV2_MODEL_FINETUNE = os.path.join(EFFICIENTNETV2_DIR, "plant_disease_efficientnetv2b0_finetune.keras")
EFFICIENTNETV2_SAVEDMODEL_DIR = os.path.join(EFFICIENTNETV2_DIR, "saved_model_efficientnetv2b0")
EFFICIENTNETV2_LABELS_PATH = os.path.join(EFFICIENTNETV2_DIR, "class_names.json")
EFFICIENTNETV2_CLASS_INDICES_PATH = os.path.join(EFFICIENTNETV2_DIR, "class_indices.json")
EFFICIENTNETV2_LOG_CSV_HEAD = os.path.join(EFFICIENTNETV2_DIR, "efficientnetv2b0_head_training_log.csv")
EFFICIENTNETV2_LOG_CSV_FT = os.path.join(EFFICIENTNETV2_DIR, "efficientnetv2b0_finetune_training_log.csv")

# Image size for EfficientNetV2B0 (RGB)
IMG_SIZE = (224, 224)

# Training parameters
BATCH_SIZE = 16
EPOCHS_HEAD = 9         # Phase 1: head only
EPOCHS_FINETUNE = 12     # Phase 2: fine-tuning
BASE_LR = 1e-3
FT_LR = 1e-5

# -----------------------------
# Clean cached EfficientNet weights (to avoid mismatched local files)
# -----------------------------
def clean_efficientnet_caches():
    keras_home = os.path.join(os.path.expanduser("~"), ".keras")

    # Remove cached EfficientNet files under .keras/models
    models_dir = os.path.join(keras_home, "models")
    if os.path.isdir(models_dir):
        removed = False
        for fn in os.listdir(models_dir):
            name = fn.lower()
            if "efficientnet" in name or "efficientnetv2" in name:
                try:
                    os.remove(os.path.join(models_dir, fn))
                    removed = True
                except Exception:
                    pass
        if removed:
            print("Removed cached EfficientNet weights from:", models_dir)

    # Remove any EfficientNet-related files under .keras/datasets
    datasets_dir = os.path.join(keras_home, "datasets")
    if os.path.isdir(datasets_dir):
        for item in os.listdir(datasets_dir):
            name = item.lower()
            if "efficientnet" in name or "efficientnetv2" in name:
                path = os.path.join(datasets_dir, item)
                try:
                    if os.path.isfile(path):
                        os.remove(path)
                    else:
                        shutil.rmtree(path)
                    print("Removed cached dataset file:", path)
                except Exception:
                    pass

# -----------------------------
# Helper functions
# -----------------------------
def get_classes(dataset_dir):
    train_dir = os.path.join(dataset_dir, "train")
    classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    classes.sort()
    return classes

def create_data_generators(dataset_dir, classes, img_size, preprocess_fn, batch_size=BATCH_SIZE):
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_fn,
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        horizontal_flip=True,
        brightness_range=[0.85, 1.15],
        zoom_range=0.15,
        fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_fn)

    train_generator = train_datagen.flow_from_directory(
        os.path.join(dataset_dir, "train"),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        classes=classes,
        shuffle=True,
        seed=SEED,
        color_mode='rgb'
    )
    val_generator = val_datagen.flow_from_directory(
        os.path.join(dataset_dir, "valid"),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        classes=classes,
        shuffle=False,
        color_mode='rgb'
    )
    return train_generator, val_generator

def plot_training_history(history, title_prefix, out_dir):
    plt.figure(figsize=(12, 4))
    # accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history.get('accuracy', []), label='Train Acc')
    plt.plot(history.history.get('val_accuracy', []), label='Val Acc')
    plt.title(f'{title_prefix} Accuracy')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True, alpha=0.3)
    # loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history.get('loss', []), label='Train Loss')
    plt.plot(history.history.get('val_loss', []), label='Val Loss')
    plt.title(f'{title_prefix} Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, f"{title_prefix.lower().replace(' ', '_')}_training_history.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved: {path}")

def save_confusion_matrix(y_true, y_pred, classes, title, out_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title, ylabel='True label', xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center", rotation_mode="anchor")
    thresh = cm.max() / 2.0 if cm.max() else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(out_path, dpi=150); plt.close()
    print(f"Saved: {out_path}")

def evaluate_and_export(model, val_generator, classes, title_prefix, model_path, savedmodel_dir, out_dir):
    steps = int(np.ceil(val_generator.samples / val_generator.batch_size))
    probs = model.predict(val_generator, steps=steps, verbose=1)
    y_pred = probs.argmax(axis=1)
    y_true = val_generator.classes

    report = classification_report(y_true, y_pred, target_names=classes)
    report_path = os.path.join(out_dir, f"{title_prefix.lower().replace(' ', '_')}_classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Saved: {report_path}")

    cm_path = os.path.join(out_dir, f"{title_prefix.lower().replace(' ', '_')}_confusion_matrix.png")
    save_confusion_matrix(y_true, y_pred, classes, f"{title_prefix} Confusion Matrix", cm_path)

    model.save(model_path)
    print(f"Saved: {model_path}")

    if os.path.isdir(savedmodel_dir):
        shutil.rmtree(savedmodel_dir)
    tf.saved_model.save(model, savedmodel_dir)
    print(f"Saved TF SavedModel: {savedmodel_dir}")

def build_efficientnetv2b0_model(num_classes, img_size):
    input_shape = (img_size[0], img_size[1], 3)
    print("Building EfficientNetV2B0 with input_shape:", input_shape)

    # Clean caches to avoid mismatched local files, then load pretrained weights
    clean_efficientnet_caches()
    tf.keras.backend.clear_session()

    base = EfficientNetV2B0(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    # Sanity checks
    assert base.input_shape[-1] == 3, f"Unexpected input channels in base: {base.input_shape}"
    print("Base model input shape:", base.input_shape)

    for layer in base.layers:
        layer.trainable = False

    x = GlobalAveragePooling2D()(base.output)
    x = Dense(768, activation='swish')(x)
    x = Dropout(0.5)(x)
    x = Dense(384, activation='swish')(x)
    x = Dropout(0.3)(x)
    out = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base.input, outputs=out)
    model.compile(optimizer=Adam(BASE_LR), loss='categorical_crossentropy', metrics=['accuracy'])

    print("Compiled model input shape:", model.input_shape)
    print("Model built with pretrained ImageNet weights.")
    return model, base

def finetune_last_layers(model, base, last_n, lr):
    for layer in base.layers[-last_n:]:
        layer.trainable = True
    model.compile(optimizer=Adam(lr), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def make_callbacks(phase, out_dir, monitor_metric="val_accuracy", patience_es=5, patience_rlrop=3, csv_path=None, ckpt_path=None):
    cbs = [
        EarlyStopping(monitor=monitor_metric, patience=patience_es, verbose=1, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=patience_rlrop, min_lr=1e-6, verbose=1)
    ]
    if ckpt_path:
        cbs.append(ModelCheckpoint(ckpt_path, monitor=monitor_metric, save_best_only=True, verbose=1))
    if csv_path:
        cbs.append(CSVLogger(csv_path))
    return cbs

if __name__ == "__main__":
    print("Starting EfficientNetV2B0 Plant Disease Classification Training (pretrained only, Keras 3 compatible)...")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Using CPU only: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")

    # Prepare classes
    print("Preparing classes...")
    classes = get_classes(DATASET_DIR)
    print(f"Detected {len(classes)} classes")
    with open(EFFICIENTNETV2_LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)

    # Data generators
    print("Creating generators for EfficientNetV2B0...")
    train_gen, val_gen = create_data_generators(
        DATASET_DIR, classes, IMG_SIZE, effv2_preprocess, BATCH_SIZE
    )
    with open(EFFICIENTNETV2_CLASS_INDICES_PATH, "w", encoding="utf-8") as f:
        json.dump(train_gen.class_indices, f, ensure_ascii=False, indent=2)

    # Build model (pretrained only)
    print("Building EfficientNetV2B0 model with pretrained weights...")
    model, base = build_efficientnetv2b0_model(len(classes), IMG_SIZE)
    if model.input_shape[-1] != 3:
        raise RuntimeError(f"Model is not 3-channel: {model.input_shape}. Aborting to avoid weight mismatch.")

    # Phase 1: head training
    callbacks_head = make_callbacks(
        phase="head",
        out_dir=EFFICIENTNETV2_DIR,
        monitor_metric="val_accuracy",
        patience_es=5,
        patience_rlrop=3,
        csv_path=EFFICIENTNETV2_LOG_CSV_HEAD,
        ckpt_path=EFFICIENTNETV2_MODEL_HEAD
    )
    print("Phase 1 training (EfficientNetV2B0 head only)...")
    history1 = model.fit(
        train_gen,
        epochs=EPOCHS_HEAD,
        validation_data=val_gen,
        callbacks=callbacks_head,
        verbose=1
    )
    plot_training_history(history1, "EfficientNetV2B0 Phase1", EFFICIENTNETV2_DIR)

    # Phase 2: fine-tuning
    print("Phase 2 fine-tuning (EfficientNetV2B0 last layers)...")
    model = finetune_last_layers(model, base, last_n=60, lr=FT_LR)

    callbacks_ft = make_callbacks(
        phase="finetune",
        out_dir=EFFICIENTNETV2_DIR,
        monitor_metric="val_accuracy",
        patience_es=3,
        patience_rlrop=2,
        csv_path=EFFICIENTNETV2_LOG_CSV_FT,
        ckpt_path=EFFICIENTNETV2_MODEL_FINETUNE
    )
    history2 = model.fit(
        train_gen,
        epochs=EPOCHS_FINETUNE,
        validation_data=val_gen,
        callbacks=callbacks_ft,
        verbose=1
    )
    plot_training_history(history2, "EfficientNetV2B0 Phase2", EFFICIENTNETV2_DIR)

    # Evaluate and export
    print("Evaluating and exporting EfficientNetV2B0 (final fine-tuned model)...")
    evaluate_and_export(
        model,
        val_gen,
        classes,
        "EfficientNetV2B0",
        EFFICIENTNETV2_MODEL_FINETUNE,
        EFFICIENTNETV2_SAVEDMODEL_DIR,
        EFFICIENTNETV2_DIR
    )

    print("\nAll training complete. Outputs saved to:")
    print(f" - {EFFICIENTNETV2_DIR}")