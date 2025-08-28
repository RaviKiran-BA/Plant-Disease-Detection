"""
import os
import json
import random
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mnv2_preprocess
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

# Optional: ensure TF determinism where possible
os.environ["TF_DETERMINISTIC_OPS"] = "1"

# -----------------------------
# Paths and config
# -----------------------------
DATASET_DIR = r"C:\Users\DELL\Downloads\New Plant Diseases Dataset(Augmented)"
HOME = os.path.expanduser("~")
OUTPUT_DIR = os.path.join(HOME, "plant_disease_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model export paths (separate files for phases)
# MobileNetV2
MOBILENET_DIR = os.path.join(OUTPUT_DIR, "mobilenetv2")
os.makedirs(MOBILENET_DIR, exist_ok=True)
MOBILENET_MODEL_HEAD = os.path.join(MOBILENET_DIR, "plant_disease_mobilenetv2_head.keras")
MOBILENET_MODEL_FINETUNE = os.path.join(MOBILENET_DIR, "plant_disease_mobilenetv2_finetune.keras")
MOBILENET_SAVEDMODEL_DIR = os.path.join(MOBILENET_DIR, "saved_model_mobilenetv2")
MOBILENET_LABELS_PATH = os.path.join(MOBILENET_DIR, "class_names.json")
MOBILENET_CLASS_INDICES_PATH = os.path.join(MOBILENET_DIR, "class_indices.json")
MOBILENET_LOG_CSV_HEAD = os.path.join(MOBILENET_DIR, "mobilenetv2_head_training_log.csv")
MOBILENET_LOG_CSV_FT = os.path.join(MOBILENET_DIR, "mobilenetv2_finetune_training_log.csv")

# Image sizes
IMG_SIZE_MOBILENET = (224, 224)

# Training parameters (CPU-friendly)
BATCH_SIZE = 16
EPOCHS_HEAD = 12          # Phase 1: head only
EPOCHS_FINETUNE = 9       # Phase 2: fine-tuning
BASE_LR = 1e-3
FT_LR = 1e-5

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
        rotation_range=35,
        width_shift_range=0.25,
        height_shift_range=0.25,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.8, 1.2],
        zoom_range=0.2,
        channel_shift_range=20.0,
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
        seed=SEED
    )
    val_generator = val_datagen.flow_from_directory(
        os.path.join(dataset_dir, "valid"),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        classes=classes,
        shuffle=False
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
    # Evaluate and get predictions
    steps = int(np.ceil(val_generator.samples / val_generator.batch_size))
    probs = model.predict(val_generator, steps=steps, verbose=1)
    y_pred = probs.argmax(axis=1)
    y_true = val_generator.classes

    # Report
    report = classification_report(y_true, y_pred, target_names=classes)
    report_path = os.path.join(out_dir, f"{title_prefix.lower().replace(' ', '_')}_classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Saved: {report_path}")

    # Confusion matrix
    cm_path = os.path.join(out_dir, f"{title_prefix.lower().replace(' ', '_')}_confusion_matrix.png")
    save_confusion_matrix(y_true, y_pred, classes, f"{title_prefix} Confusion Matrix", cm_path)

    # Save models
    model.save(model_path)
    print(f"Saved: {model_path}")
    # SavedModel export
    # Remove existing SavedModel dir to avoid version conflicts
    if os.path.isdir(savedmodel_dir):
        import shutil
        shutil.rmtree(savedmodel_dir)
    tf.saved_model.save(model, savedmodel_dir)
    print(f"Saved TF SavedModel: {savedmodel_dir}")

def build_mobilenet_model(num_classes, img_size):
    base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_size[0], img_size[1], 3))
    for layer in base.layers:
        layer.trainable = False
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    out = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base.input, outputs=out)
    model.compile(optimizer=Adam(BASE_LR), loss='categorical_crossentropy', metrics=['accuracy'])
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
    # ---------------------------------
    # Prepare classes (shared)
    # ---------------------------------
    print("Preparing classes...")
    classes = get_classes(DATASET_DIR)
    print(f"Detected {len(classes)} classes")

    # Save class names in model folder
    with open(MOBILENET_LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)

    # ---------------------------------
    # MobileNetV2 training
    # ---------------------------------
    print("Creating generators for MobileNetV2...")
    train_gen_m, val_gen_m = create_data_generators(DATASET_DIR, classes, IMG_SIZE_MOBILENET, mnv2_preprocess, BATCH_SIZE)

    # Save class indices for reproducibility/inference
    with open(MOBILENET_CLASS_INDICES_PATH, "w", encoding="utf-8") as f:
        json.dump(train_gen_m.class_indices, f, ensure_ascii=False, indent=2)

    print("Building MobileNetV2 model...")
    model_m, base_m = build_mobilenet_model(len(classes), IMG_SIZE_MOBILENET)

    # Callbacks for phase 1 (head)
    callbacks_m_head = make_callbacks(
        phase="head",
        out_dir=MOBILENET_DIR,
        monitor_metric="val_accuracy",
        patience_es=5,
        patience_rlrop=3,
        csv_path=MOBILENET_LOG_CSV_HEAD,
        ckpt_path=MOBILENET_MODEL_HEAD
    )

    print("Phase 1 training (MobileNetV2 head only)...")
    history_m1 = model_m.fit(
        train_gen_m,
        epochs=EPOCHS_HEAD,
        validation_data=val_gen_m,
        callbacks=callbacks_m_head,
        verbose=1
    )
    plot_training_history(history_m1, "MobileNetV2 Phase1", MOBILENET_DIR)

    print("Phase 2 fine-tuning (MobileNetV2 last layers)...")
    model_m = finetune_last_layers(model_m, base_m, last_n=40, lr=FT_LR)

    # Recreate callbacks for phase 2 (reset state, maybe shorter patience)
    callbacks_m_ft = make_callbacks(
        phase="finetune",
        out_dir=MOBILENET_DIR,
        monitor_metric="val_accuracy",
        patience_es=3,            # slightly shorter patience for FT
        patience_rlrop=2,
        csv_path=MOBILENET_LOG_CSV_FT,
        ckpt_path=MOBILENET_MODEL_FINETUNE
    )

    history_m2 = model_m.fit(
        train_gen_m,
        epochs=EPOCHS_FINETUNE,
        validation_data=val_gen_m,
        callbacks=callbacks_m_ft,
        verbose=1
    )
    plot_training_history(history_m2, "MobileNetV2 Phase2", MOBILENET_DIR)

    print("Evaluating and exporting MobileNetV2 (final fine-tuned model)...")
    evaluate_and_export(
        model_m,
        val_gen_m,
        classes,
        "MobileNetV2",
        MOBILENET_MODEL_FINETUNE,
        MOBILENET_SAVEDMODEL_DIR,
        MOBILENET_DIR
    )
    print("MobileNetV2 training complete.")

    print(f"\nTraining complete. Outputs saved to: {MOBILENET_DIR}")
    """
import os
import json
import random
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

from tensorflow.keras.applications import MobileNetV2, EfficientNetB3
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mnv2_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as effb3_preprocess
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

# Optional: ensure TF determinism where possible
os.environ["TF_DETERMINISTIC_OPS"] = "1"

# -----------------------------
# Paths and config
# -----------------------------
DATASET_DIR = r"C:\Users\DELL\Downloads\New Plant Diseases Dataset(Augmented)"
HOME = os.path.expanduser("~")
OUTPUT_DIR = os.path.join(HOME, "plant_disease_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model export paths (separate files for phases)
# MobileNetV2
MOBILENET_DIR = os.path.join(OUTPUT_DIR, "mobilenetv2")
os.makedirs(MOBILENET_DIR, exist_ok=True)
MOBILENET_MODEL_HEAD = os.path.join(MOBILENET_DIR, "plant_disease_mobilenetv2_head.keras")
MOBILENET_MODEL_FINETUNE = os.path.join(MOBILENET_DIR, "plant_disease_mobilenetv2_finetune.keras")
MOBILENET_SAVEDMODEL_DIR = os.path.join(MOBILENET_DIR, "saved_model_mobilenetv2")
MOBILENET_LABELS_PATH = os.path.join(MOBILENET_DIR, "class_names.json")
MOBILENET_CLASS_INDICES_PATH = os.path.join(MOBILENET_DIR, "class_indices.json")
MOBILENET_LOG_CSV_HEAD = os.path.join(MOBILENET_DIR, "mobilenetv2_head_training_log.csv")
MOBILENET_LOG_CSV_FT = os.path.join(MOBILENET_DIR, "mobilenetv2_finetune_training_log.csv")

# EfficientNetB3
EFFICIENTNET_DIR = os.path.join(OUTPUT_DIR, "efficientnetb3")
os.makedirs(EFFICIENTNET_DIR, exist_ok=True)
EFFICIENTNET_MODEL_HEAD = os.path.join(EFFICIENTNET_DIR, "plant_disease_efficientnetb3_head.keras")
EFFICIENTNET_MODEL_FINETUNE = os.path.join(EFFICIENTNET_DIR, "plant_disease_efficientnetb3_finetune.keras")
EFFICIENTNET_SAVEDMODEL_DIR = os.path.join(EFFICIENTNET_DIR, "saved_model_efficientnetb3")
EFFICIENTNET_LABELS_PATH = os.path.join(EFFICIENTNET_DIR, "class_names.json")
EFFICIENTNET_CLASS_INDICES_PATH = os.path.join(EFFICIENTNET_DIR, "class_indices.json")
EFFICIENTNET_LOG_CSV_HEAD = os.path.join(EFFICIENTNET_DIR, "efficientnetb3_head_training_log.csv")
EFFICIENTNET_LOG_CSV_FT = os.path.join(EFFICIENTNET_DIR, "efficientnetb3_finetune_training_log.csv")

# Image sizes
IMG_SIZE_MOBILENET = (224, 224)
IMG_SIZE_EFFICIENTNET = (300, 300)  # common for B3

# Training parameters (CPU-friendly)
BATCH_SIZE = 16
EPOCHS_HEAD = 12          # Phase 1: head only
EPOCHS_FINETUNE = 9       # Phase 2: fine-tuning
BASE_LR = 1e-3
FT_LR = 1e-5

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
        rotation_range=35,
        width_shift_range=0.25,
        height_shift_range=0.25,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.8, 1.2],
        zoom_range=0.2,
        channel_shift_range=20.0,
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
        seed=SEED
    )
    val_generator = val_datagen.flow_from_directory(
        os.path.join(dataset_dir, "valid"),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        classes=classes,
        shuffle=False
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
    # Evaluate and get predictions
    steps = int(np.ceil(val_generator.samples / val_generator.batch_size))
    probs = model.predict(val_generator, steps=steps, verbose=1)
    y_pred = probs.argmax(axis=1)
    y_true = val_generator.classes

    # Report
    report = classification_report(y_true, y_pred, target_names=classes)
    report_path = os.path.join(out_dir, f"{title_prefix.lower().replace(' ', '_')}_classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Saved: {report_path}")

    # Confusion matrix
    cm_path = os.path.join(out_dir, f"{title_prefix.lower().replace(' ', '_')}_confusion_matrix.png")
    save_confusion_matrix(y_true, y_pred, classes, f"{title_prefix} Confusion Matrix", cm_path)

    # Save models
    model.save(model_path)
    print(f"Saved: {model_path}")
    # SavedModel export
    # Remove existing SavedModel dir to avoid version conflicts
    if os.path.isdir(savedmodel_dir):
        import shutil
        shutil.rmtree(savedmodel_dir)
    tf.saved_model.save(model, savedmodel_dir)
    print(f"Saved TF SavedModel: {savedmodel_dir}")

def build_mobilenet_model(num_classes, img_size):
    base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_size[0], img_size[1], 3))
    for layer in base.layers:
        layer.trainable = False
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    out = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base.input, outputs=out)
    model.compile(optimizer=Adam(BASE_LR), loss='categorical_crossentropy', metrics=['accuracy'])
    return model, base

def build_efficientnetb3_model(num_classes, img_size):
    base = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(img_size[0], img_size[1], 3))
    for layer in base.layers:
        layer.trainable = False
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='swish')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='swish')(x)
    x = Dropout(0.3)(x)
    out = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base.input, outputs=out)
    model.compile(optimizer=Adam(BASE_LR), loss='categorical_crossentropy', metrics=['accuracy'])
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
    # ---------------------------------
    # Prepare classes (shared)
    # ---------------------------------
    print("Preparing classes...")
    classes = get_classes(DATASET_DIR)
    print(f"Detected {len(classes)} classes")

    # Save class names in both model folders (for convenience)
    with open(MOBILENET_LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)
    with open(EFFICIENTNET_LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)

    # ---------------------------------
    # MobileNetV2 training
    # ---------------------------------
    print("Creating generators for MobileNetV2...")
    train_gen_m, val_gen_m = create_data_generators(DATASET_DIR, classes, IMG_SIZE_MOBILENET, mnv2_preprocess, BATCH_SIZE)

    # Save class indices for reproducibility/inference
    with open(MOBILENET_CLASS_INDICES_PATH, "w", encoding="utf-8") as f:
        json.dump(train_gen_m.class_indices, f, ensure_ascii=False, indent=2)

    print("Building MobileNetV2 model...")
    model_m, base_m = build_mobilenet_model(len(classes), IMG_SIZE_MOBILENET)

    # Callbacks for phase 1 (head)
    callbacks_m_head = make_callbacks(
        phase="head",
        out_dir=MOBILENET_DIR,
        monitor_metric="val_accuracy",
        patience_es=5,
        patience_rlrop=3,
        csv_path=MOBILENET_LOG_CSV_HEAD,
        ckpt_path=MOBILENET_MODEL_HEAD
    )

    print("Phase 1 training (MobileNetV2 head only)...")
    history_m1 = model_m.fit(
        train_gen_m,
        epochs=EPOCHS_HEAD,
        validation_data=val_gen_m,
        callbacks=callbacks_m_head,
        verbose=1
    )
    plot_training_history(history_m1, "MobileNetV2 Phase1", MOBILENET_DIR)

    print("Phase 2 fine-tuning (MobileNetV2 last layers)...")
    model_m = finetune_last_layers(model_m, base_m, last_n=40, lr=FT_LR)

    # Recreate callbacks for phase 2 (reset state, maybe shorter patience)
    callbacks_m_ft = make_callbacks(
        phase="finetune",
        out_dir=MOBILENET_DIR,
        monitor_metric="val_accuracy",
        patience_es=3,            # slightly shorter patience for FT
        patience_rlrop=2,
        csv_path=MOBILENET_LOG_CSV_FT,
        ckpt_path=MOBILENET_MODEL_FINETUNE
    )

    history_m2 = model_m.fit(
        train_gen_m,
        epochs=EPOCHS_FINETUNE,
        validation_data=val_gen_m,
        callbacks=callbacks_m_ft,
        verbose=1
    )
    plot_training_history(history_m2, "MobileNetV2 Phase2", MOBILENET_DIR)

    print("Evaluating and exporting MobileNetV2 (final fine-tuned model)...")
    evaluate_and_export(
        model_m,
        val_gen_m,
        classes,
        "MobileNetV2",
        MOBILENET_MODEL_FINETUNE,
        MOBILENET_SAVEDMODEL_DIR,
        MOBILENET_DIR
    )
    print("MobileNetV2 training complete.")

    # Free resources before next model
    tf.keras.backend.clear_session()

    # ---------------------------------
    # EfficientNetB3 training
    # ---------------------------------
    print("Creating generators for EfficientNetB3...")
    train_gen_e, val_gen_e = create_data_generators(DATASET_DIR, classes, IMG_SIZE_EFFICIENTNET, effb3_preprocess, BATCH_SIZE)

    # Save class indices for reproducibility/inference
    with open(EFFICIENTNET_CLASS_INDICES_PATH, "w", encoding="utf-8") as f:
        json.dump(train_gen_e.class_indices, f, ensure_ascii=False, indent=2)

    print("Building EfficientNetB3 model...")
    model_e, base_e = build_efficientnetb3_model(len(classes), IMG_SIZE_EFFICIENTNET)

    callbacks_e_head = make_callbacks(
        phase="head",
        out_dir=EFFICIENTNET_DIR,
        monitor_metric="val_accuracy",
        patience_es=5,
        patience_rlrop=3,
        csv_path=EFFICIENTNET_LOG_CSV_HEAD,
        ckpt_path=EFFICIENTNET_MODEL_HEAD
    )

    print("Phase 1 training (EfficientNetB3 head only)...")
    history_e1 = model_e.fit(
        train_gen_e,
        epochs=EPOCHS_HEAD,
        validation_data=val_gen_e,
        callbacks=callbacks_e_head,
        verbose=1
    )
    plot_training_history(history_e1, "EfficientNetB3 Phase1", EFFICIENTNET_DIR)

    print("Phase 2 fine-tuning (EfficientNetB3 last layers)...")
    model_e = finetune_last_layers(model_e, base_e, last_n=60, lr=FT_LR)

    callbacks_e_ft = make_callbacks(
        phase="finetune",
        out_dir=EFFICIENTNET_DIR,
        monitor_metric="val_accuracy",
        patience_es=3,
        patience_rlrop=2,
        csv_path=EFFICIENTNET_LOG_CSV_FT,
        ckpt_path=EFFICIENTNET_MODEL_FINETUNE
    )

    history_e2 = model_e.fit(
        train_gen_e,
        epochs=EPOCHS_FINETUNE,
        validation_data=val_gen_e,
        callbacks=callbacks_e_ft,
        verbose=1
    )
    plot_training_history(history_e2, "EfficientNetB3 Phase2", EFFICIENTNET_DIR)

    print("Evaluating and exporting EfficientNetB3 (final fine-tuned model)...")
    evaluate_and_export(
        model_e,
        val_gen_e,
        classes,
        "EfficientNetB3",
        EFFICIENTNET_MODEL_FINETUNE,
        EFFICIENTNET_SAVEDMODEL_DIR,
        EFFICIENTNET_DIR
    )
    print("EfficientNetB3 training complete.")

    print("\nAll training complete. Outputs saved to:")
    print(f" - {MOBILENET_DIR}")
    print(f" - {EFFICIENTNET_DIR}")