# Extract
import zipfile
import os

# Path to your zip file
zip_path = '/content/PlantAI.zip'
extract_path = '/content/Extract'

# Unzipping the file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# Check if the extraction worked
print(os.listdir(extract_path))

# Image data generator
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Image dimensions and batch size
img_width, img_height = 150, 150
batch_size = 32

# Image Data Generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(rescale=1./255)

# Ensure the paths to your data directories are correct
train_dir = '/content/Extract/Image data/train'
val_dir = '/content/Extract/Image data/validation'
test_dir = '/content/Extract/Image data/test'

# Train data generator
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

# Validation data generator
validation_generator = test_datagen.flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

# Test data generator
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # Generally set to False for test data
)

# Check for missing images
import os

# Path to your dataset directories
extract_path = '/content/Extract'
train_dir = os.path.join(extract_path, 'Image data', 'train')
val_dir = os.path.join(extract_path, 'Image data', 'validation')
test_dir = os.path.join(extract_path, 'Image data', 'test')

# Expected counts for train, validation, and test folders
expected_counts = {
    'Image data/train': 31032,
    'Image data/validation': 8137,
    'Image data/test': 4762
}

# Function to count the number of images in a given directory
def count_images_in_directory(directory):
    total_images = 0
    for root, _, files in os.walk(directory):
        total_images += len([f for f in files if os.path.isfile(os.path.join(root, f))])
    return total_images

# Checking each directory for missing images
missing_images_report = {}

for folder, expected_count in expected_counts.items():
    folder_path = os.path.join(extract_path, folder)
    if not os.path.exists(folder_path):
        print(f"Warning: {folder_path} does not exist.")
        missing_images_report[folder] = f"Directory not found"
        continue

    # Count actual images
    actual_count = count_images_in_directory(folder_path)
    missing_count = expected_count - actual_count

    # Report if images are missing
    if missing_count > 0:
        missing_images_report[folder] = f"{missing_count} images are missing"
    elif missing_count < 0:
        missing_images_report[folder] = f"{abs(missing_count)} extra images found"
    else:
        missing_images_report[folder] = "All images are present"

# Display missing images report
print("Missing Images Report:")
for folder, report in missing_images_report.items():
    print(f"{folder:25}: {report}")

# check for total classes
import tensorflow as tf

# Function to load a dataset from a directory
def load_dataset(directory, batch_size=32, image_size=(224, 224), shuffle=True):
    return tf.keras.preprocessing.image_dataset_from_directory(
        directory,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=123 if shuffle else None  # Set seed only if shuffling
    )

# Load the training and validation datasets
training_set = load_dataset("/content/Extract/Image data/train", shuffle=True)
validation_set = load_dataset("/content/Extract/Image data/validation", shuffle=False)

# Retrieve class names from both datasets
class_names_train = training_set.class_names
class_names_val = validation_set.class_names

# Combine the class names from both training and validation sets, and remove duplicates
all_class_names = set(class_names_train).union(set(class_names_val))

# Separate classes into 'healthy' and 'diseases'
healthy_classes = [class_name for class_name in all_class_names if 'healthy' in class_name.lower()]
disease_classes = [class_name for class_name in all_class_names if 'healthy' not in class_name.lower()]

# Print the classes and total counts
print("Healthy Classes:")
for healthy_class in sorted(healthy_classes):
    print(f"- {healthy_class}")

print("\nDisease Classes:")
for disease_class in sorted(disease_classes):
    print(f"- {disease_class}")

print(f"\nTotal number of healthy classes: {len(healthy_classes)}")
print(f"Total number of disease classes: {len(disease_classes)}")
print(f"Total number of unique classes: {len(all_class_names)}")

# Pre-processing
import os
import cv2
import numpy as np
from pathlib import Path
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_image(image_path):
    """Load image from a file."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found or invalid at path: {image_path}")
    return image

def remove_background(image):
    """Remove background using a mask based on a color threshold."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    foreground = cv2.bitwise_and(image, image, mask=mask)
    return foreground

def apply_histogram_equalization(image):
    """Apply histogram equalization to enhance contrast."""
    return cv2.equalizeHist(image)

def resize_image(image, target_size):
    """Resize image to a fixed size using INTER_AREA for better quality when resizing."""
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

def crop_image(image, crop_size=(224, 224)):
    """Crop the image to the center with a specified crop size."""
    h, w = image.shape[:2]
    start_x = w // 2 - (crop_size[0] // 2)
    start_y = h // 2 - (crop_size[1] // 2)
    return image[start_y:start_y+crop_size[1], start_x:start_x+crop_size[0]]

def normalize_pixel_values(image):
    """Normalize pixel values to the range [0, 1]."""
    return image.astype(np.float32) / 255.0

def preprocess_image(image_path, output_path, target_size=(224, 224)):
    """Preprocess the image by removing background, applying histogram equalization to each channel, cropping, resizing, and normalizing pixel values."""
    image = load_image(image_path)

    # Step 1: Background Removal
    image = remove_background(image)

    # Step 2: Histogram Equalization on each color channel
    channels = cv2.split(image)
    equalized_channels = [apply_histogram_equalization(channel) for channel in channels]
    image = cv2.merge(equalized_channels)

    # Step 3: Cropping Image to the Center
    image = crop_image(image, crop_size=target_size)

    # Step 4: Resizing Image to the Target Size
    image = resize_image(image, target_size)

    # Step 5: Normalizing Pixel Values (scale pixel values to [0, 1])
    image_normalized = normalize_pixel_values(image)

    # Save the preprocessed image in the output directory
    cv2.imwrite(str(output_path), (image_normalized * 255).astype(np.uint8))
    return image_normalized

def is_valid_image(file_path):
    """Check if a file is a valid image."""
    try:
        img = cv2.imread(str(file_path))
        return img is not None
    except Exception:
        return False

def preprocess_dataset(input_folder, output_folder, target_size=(224, 224)):
    """Preprocess all images in the dataset by applying background removal, histogram equalization, cropping, resizing, and pixel normalization."""
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    image_extensions = {'.jpg', '.jpeg', '.png'}

    total_images = 0
    processed_images = 0

    for subdir, dirs, files in os.walk(input_path):
        for file in files:
            if file.lower().endswith(tuple(image_extensions)):
                total_images += 1

                img_path = Path(subdir) / file
                relative_path = img_path.relative_to(input_path)
                output_img_path = output_path / relative_path

                # Create necessary subdirectories in the output path
                output_img_path.parent.mkdir(parents=True, exist_ok=True)

                try:
                    if is_valid_image(img_path):
                        preprocess_image(str(img_path), str(output_img_path), target_size)
                        processed_images += 1

                        if processed_images % 10 == 0:
                            logging.info(f"Processed {processed_images}/{total_images}: {relative_path}")
                    else:
                        logging.warning(f"Skipped invalid image: {relative_path}")
                except Exception as e:
                    logging.error(f"Error processing {relative_path}: {str(e)}")

    logging.info(f"Total images found: {total_images}")
    logging.info(f"Total images processed: {processed_images}")

    # Print the number of preprocessed images
    print(f"Total number of images preprocessed: {processed_images}")

    if processed_images == 0:
        logging.warning("No images were processed. Check your input folder and image files.")

if __name__ == "__main__":
    input_folder = "/content/Extract/Image data/train"  # Adjust this to your main input folder
    output_folder = "/content/Pre-Processed_Images/Preprocessed_Images_Train"
    target_size = (224, 224)

    start_time = time.time()

    logging.info(f"Starting preprocessing. Input folder: {input_folder}")

    preprocess_dataset(input_folder, output_folder, target_size)

    end_time = time.time()
    logging.info(f"Preprocessing completed. Total time: {end_time - start_time:.2f} seconds")

import os
import cv2
import numpy as np
from pathlib import Path
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_image(image_path):
    """Load image from a file."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found or invalid at path: {image_path}")
    return image

def remove_background(image):
    """Remove background using a mask based on a color threshold."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    foreground = cv2.bitwise_and(image, image, mask=mask)
    return foreground

def apply_histogram_equalization(image):
    """Apply histogram equalization to enhance contrast."""
    return cv2.equalizeHist(image)

def resize_image(image, target_size):
    """Resize image to a fixed size using INTER_AREA for better quality when resizing."""
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

def crop_image(image, crop_size=(224, 224)):
    """Crop the image to the center with a specified crop size."""
    h, w = image.shape[:2]
    start_x = w // 2 - (crop_size[0] // 2)
    start_y = h // 2 - (crop_size[1] // 2)
    return image[start_y:start_y+crop_size[1], start_x:start_x+crop_size[0]]

def normalize_pixel_values(image):
    """Normalize pixel values to the range [0, 1]."""
    return image.astype(np.float32) / 255.0

def preprocess_image(image_path, output_path, target_size=(224, 224)):
    """Preprocess the image by removing background, applying histogram equalization to each channel, cropping, resizing, and normalizing pixel values."""
    image = load_image(image_path)

    # Step 1: Background Removal
    image = remove_background(image)

    # Step 2: Histogram Equalization on each color channel
    channels = cv2.split(image)
    equalized_channels = [apply_histogram_equalization(channel) for channel in channels]
    image = cv2.merge(equalized_channels)

    # Step 3: Cropping Image to the Center
    image = crop_image(image, crop_size=target_size)

    # Step 4: Resizing Image to the Target Size
    image = resize_image(image, target_size)

    # Step 5: Normalizing Pixel Values (scale pixel values to [0, 1])
    image_normalized = normalize_pixel_values(image)

    # Save the preprocessed image in the output directory
    cv2.imwrite(str(output_path), (image_normalized * 255).astype(np.uint8))
    return image_normalized

def is_valid_image(file_path):
    """Check if a file is a valid image."""
    try:
        img = cv2.imread(str(file_path))
        return img is not None
    except Exception:
        return False

def preprocess_dataset(input_folder, output_folder, target_size=(224, 224)):
    """Preprocess all images in the dataset by applying background removal, histogram equalization, cropping, resizing, and pixel normalization."""
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    image_extensions = {'.jpg', '.jpeg', '.png'}

    total_images = 0
    processed_images = 0

    for subdir, dirs, files in os.walk(input_path):
        for file in files:
            if file.lower().endswith(tuple(image_extensions)):
                total_images += 1

                img_path = Path(subdir) / file
                relative_path = img_path.relative_to(input_path)
                output_img_path = output_path / relative_path

                # Create necessary subdirectories in the output path
                output_img_path.parent.mkdir(parents=True, exist_ok=True)

                try:
                    if is_valid_image(img_path):
                        preprocess_image(str(img_path), str(output_img_path), target_size)
                        processed_images += 1

                        if processed_images % 10 == 0:
                            logging.info(f"Processed {processed_images}/{total_images}: {relative_path}")
                    else:
                        logging.warning(f"Skipped invalid image: {relative_path}")
                except Exception as e:
                    logging.error(f"Error processing {relative_path}: {str(e)}")

    logging.info(f"Total images found: {total_images}")
    logging.info(f"Total images processed: {processed_images}")

    # Print the number of preprocessed images
    print(f"Total number of images preprocessed: {processed_images}")

    if processed_images == 0:
        logging.warning("No images were processed. Check your input folder and image files.")

if __name__ == "__main__":
    input_folder = "/content/Extract/Image data/validation"  # Adjust this to your main input folder
    output_folder = "/content/Pre-Processed_Images/Preprocessed_Images_Validation"
    target_size = (224, 224)

    start_time = time.time()

    logging.info(f"Starting preprocessing. Input folder: {input_folder}")

    preprocess_dataset(input_folder, output_folder, target_size)

    end_time = time.time()
    logging.info(f"Preprocessing completed. Total time: {end_time - start_time:.2f} seconds") # 1m 12s

# Normalize the images
import tensorflow as tf
from tensorflow.keras.layers import Rescaling
import matplotlib.pyplot as plt

# Function to apply normalization using the Rescaling layer
def normalize_dataset(dataset):
    normalization_layer = Rescaling(1.0 / 255)
    normalized_dataset = dataset.map(lambda x, y: (normalization_layer(x), y))
    return normalized_dataset

# Function to load a dataset from a directory
def load_dataset(directory, batch_size=32, image_size=(224, 224), shuffle=True):
    return tf.keras.preprocessing.image_dataset_from_directory(
        directory,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=123 if shuffle else None  # Set seed only if shuffling
    )

# Function to display a sample of images from a dataset with their disease names
def display_sample_images(dataset, class_names, title, num_images=1):  # Set num_images to 1
    plt.figure(figsize=(5, 5))  # Set the figure size

    for images, labels in dataset.take(1):  # Take one batch from the dataset
        for i in range(num_images):  # Display 'num_images' images
            ax = plt.subplot(1, 1, 1)  # 1 row, 1 column (for 1 image)
            plt.imshow((images[i].numpy() * 255.0).astype("uint8"))
            plt.title(f"{class_names[labels[i].numpy()]}")  # Show the disease name
            plt.axis("off")  # Turn off axis display

    plt.suptitle(title, fontsize=16)  # Add a title to the figure
    plt.show()

# Load and normalize the training dataset
training_set = load_dataset("/content/Pre-Processed_Images/Preprocessed_Images_Train", shuffle=True)
normalized_training_set = normalize_dataset(training_set)
class_names_train = training_set.class_names

# Load and normalize the validation dataset
validation_set = load_dataset("/content/Pre-Processed_Images/Preprocessed_Images_Validation", shuffle=False)
normalized_validation_set = normalize_dataset(validation_set)
class_names_val = validation_set.class_names

# Display one normalized image from the training set
display_sample_images(normalized_training_set, class_names_train, title="Normalized Training Set Image", num_images=1)

# Display one normalized image from the validation set
display_sample_images(normalized_validation_set, class_names_val, title="Normalized Validation Set Image", num_images=1)

# Segmentation
import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def preprocess_image(image):
    # Resizing and Scaling
    img_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Sharpening
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    img_sharpened = cv2.filter2D(img_resized, -1, kernel)

    # Brightening and Contrast Adjustment
    img_bright = cv2.convertScaleAbs(img_resized, alpha=1.2, beta=10)

    # Watershed Segmentation
    img_watershed = watershed_segmentation(img_resized.copy())

    # Canny Edge Detection
    img_edges = canny_edge_detection(img_resized)

    # Adaptive Thresholding
    img_adaptive_thresh = adaptive_thresholding(img_resized)

    return {
        'original': img_resized,
        'sharpened': img_sharpened,
        'brightened': img_bright,
        'watershed': img_watershed,
        'edges': img_edges,
        'adaptive_thresh': img_adaptive_thresh
    }

def watershed_segmentation(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(image, markers)
    image[markers == -1] = [255, 0, 0]
    return image

def canny_edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)  # Convert back to RGB for consistency

def adaptive_thresholding(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return cv2.cvtColor(adaptive_thresh, cv2.COLOR_GRAY2RGB)  # Convert back to RGB for consistency

def get_image_files(directory):
    image_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
                image_files.append(os.path.join(root, file))
    return image_files

def process_images(input_dir, output_dir):
    image_files = get_image_files(input_dir)

    sample_processed = None

    for file in tqdm(image_files, desc=f"Processing images in {os.path.basename(input_dir)}"):
        img = cv2.imread(file)
        if img is None:
            print(f"Warning: Unable to read image {file}")
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        processed = preprocess_image(img_rgb)

        # Save the first processed image only for display
        if sample_processed is None:
            sample_processed = processed

        for key, processed_img in processed.items():
            # Create a subdirectory for each type of processed image
            subdir = os.path.join(output_dir, key)
            os.makedirs(subdir, exist_ok=True)

            # Preserve the original directory structure
            rel_path = os.path.relpath(file, input_dir)
            output_path = os.path.join(subdir, f"{os.path.splitext(rel_path)[0]}.jpg")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            cv2.imwrite(output_path, cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR))

    return len(image_files), sample_processed

def display_results(train_sample, val_sample):
    # Create a grid layout for displaying all processed images
    num_images = 6  # Number of processed images per sample (original, sharpened, brightened, watershed, edges, adaptive_thresh)
    fig, axs = plt.subplots(2, num_images, figsize=(15, 6))
    fig.suptitle("Processed Images: Training (First Row) and Validation (Second Row)")

    # Display training sample processed images
    if train_sample is not None:
        for i, (key, img) in enumerate(train_sample.items()):
            axs[0, i].imshow(img)
            axs[0, i].set_title(f"Train: {key.capitalize()}")
            axs[0, i].axis('off')
    else:
        axs[0, 0].set_title("No Training Sample")
        axs[0, 0].axis('off')

    # Display validation sample processed images
    if val_sample is not None:
        for i, (key, img) in enumerate(val_sample.items()):
            axs[1, i].imshow(img)
            axs[1, i].set_title(f"Val: {key.capitalize()}")
            axs[1, i].axis('off')
    else:
        axs[1, 0].set_title("No Validation Sample")
        axs[1, 0].axis('off')

    plt.tight_layout()
    plt.show()

def process_dataset(train_dir, val_dir, output_dir):
    # Create output directories
    train_output_dir = os.path.join(output_dir, 'train_seg')
    val_output_dir = os.path.join(output_dir, 'val_seg')

    # Process training images
    train_count, train_sample = process_images(train_dir, train_output_dir)

    # Check validation directory
    val_image_files = get_image_files(val_dir)
    if not val_image_files:
        print("Warning: No images found in the validation directory.")
        val_sample = None  # No validation images to process
    else:
        # Process validation images
        val_count, val_sample = process_images(val_dir, val_output_dir)

    print(f"Processed {train_count} training images.")
    if val_sample:
        print(f"Processed {val_count} validation images.")
    else:
        print("No validation images processed.")

    print(f"Processed training images are saved in: {train_output_dir}")
    print(f"Processed validation images are saved in: {val_output_dir}")
    print("Each type of processed image is saved in its own subdirectory within these folders.")

    # Display sample results
    display_results(train_sample, val_sample)

# Usage
train_directory = '/content/Pre-Processed_Images/Preprocessed_Images_Train'
val_directory = '/content/Pre-Processed_Images/Preprocessed_Images_Validation'
output_directory = '/content/Segmented_Images'

process_dataset(train_directory, val_directory, output_directory)

# Feature Extraction
import os
import cv2
import numpy as np
from skimage.feature import hog
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import random

def get_image_files(directory, limit=None):
    image_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
                image_files.append(os.path.join(root, file))
                if limit and len(image_files) >= limit:
                    return image_files
    return image_files

def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_channels = np.mean(image, axis=(0, 1))
    edges = cv2.Canny(gray, 100, 200)
    hog_features, hog_image = hog(gray, visualize=True, block_norm='L2-Hys', pixels_per_cell=(16, 16))

    return {
        'gray': gray,
        'mean_channels': mean_channels,
        'edges': edges,
        'hog_features': hog_features,
        'hog_image': hog_image
    }

def save_images(features, base_name, save_directory):
    # Save each feature as a separate .jpg image
    cv2.imwrite(os.path.join(save_directory, f"{base_name}_gray.jpg"), features['gray'])
    cv2.imwrite(os.path.join(save_directory, f"{base_name}_edges.jpg"), features['edges'])
    cv2.imwrite(os.path.join(save_directory, f"{base_name}_hog.jpg"), (features['hog_image'] * 255).astype(np.uint8))

def process_image(args):
    image_path, save_directory = args
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None

        features = extract_features(image)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        save_images(features, base_name, save_directory)
        return save_directory
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

def process_images_batch(image_files, save_directory, batch_size=1000, max_workers=None):
    os.makedirs(save_directory, exist_ok=True)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        args = [(img_path, save_directory) for img_path in image_files]
        results = list(tqdm(executor.map(process_image, args, chunksize=batch_size), total=len(image_files), desc="Processing images"))
    return results

def display_features_and_numerical_outputs(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image at {image_path}")
        return

    features = extract_features(image)

    # Display extracted features
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Feature Extraction Results')

    axs[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs[0, 0].set_title('Original Image')
    axs[0, 0].axis('off')

    axs[0, 1].imshow(features['gray'], cmap='gray')
    axs[0, 1].set_title('Grayscale')
    axs[0, 1].axis('off')

    axs[0, 2].bar(['Blue', 'Green', 'Red'], features['mean_channels'])
    axs[0, 2].set_title('Mean Channel Values')

    axs[1, 0].imshow(features['edges'], cmap='gray')
    axs[1, 0].set_title('Edge Features')
    axs[1, 0].axis('off')

    axs[1, 1].imshow(features['hog_image'], cmap='gray')
    axs[1, 1].set_title('HOG Features')
    axs[1, 1].axis('off')

    axs[1, 2].axis('off')

    plt.tight_layout()
    plt.show()

    # Print numerical outputs
    print("Featured Extracted:")
    print(f"Mean Channel Values: {features['mean_channels']}")
    print(f"Number of Edge Pixels: {np.sum(features['edges'] > 0)}")
    print(f"HOG Features Shape: {features['hog_features'].shape}")
    print(f"HOG Features Mean: {np.mean(features['hog_features'])}")
    print(f"HOG Features Standard Deviation: {np.std(features['hog_features'])}")

if __name__ == "__main__":
    limit = 100  # Set the limit as desired
    train_directory = '/content/Pre-Processed_Images/Preprocessed_Images_Train'
    train_output_directory = '/content/Feature_Extracted_Images/train'

    # Process all images and save features
    image_files = get_image_files(train_directory, limit=limit)
    if image_files:
        print(f"Processing {len(image_files)} training images...")
        process_images_batch(image_files, train_output_directory)
        print("Feature extraction and saving completed as .jpg images.")

        # Randomly select one image from the processed list to display features
        sample_image = random.choice(image_files)
        print(f"Displaying features for randomly selected image: {sample_image}")
        display_features_and_numerical_outputs(sample_image)
    else:
        print("No images found in the training directory.")

import os
import cv2
import numpy as np
from skimage.feature import hog
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import random

def get_image_files(directory, limit=None):
    image_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
                image_files.append(os.path.join(root, file))
                if limit and len(image_files) >= limit:
                    return image_files
    return image_files

def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_channels = np.mean(image, axis=(0, 1))
    edges = cv2.Canny(gray, 100, 200)
    hog_features, hog_image = hog(gray, visualize=True, block_norm='L2-Hys', pixels_per_cell=(16, 16))

    return {
        'gray': gray,
        'mean_channels': mean_channels,
        'edges': edges,
        'hog_features': hog_features,
        'hog_image': hog_image
    }

def save_images(features, base_name, save_directory):
    # Save each feature as a separate .jpg image
    cv2.imwrite(os.path.join(save_directory, f"{base_name}_gray.jpg"), features['gray'])
    cv2.imwrite(os.path.join(save_directory, f"{base_name}_edges.jpg"), features['edges'])
    cv2.imwrite(os.path.join(save_directory, f"{base_name}_hog.jpg"), (features['hog_image'] * 255).astype(np.uint8))

def process_image(args):
    image_path, save_directory = args
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None

        features = extract_features(image)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        save_images(features, base_name, save_directory)
        return save_directory
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

def process_images_batch(image_files, save_directory, batch_size=1000, max_workers=None):
    os.makedirs(save_directory, exist_ok=True)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        args = [(img_path, save_directory) for img_path in image_files]
        results = list(tqdm(executor.map(process_image, args, chunksize=batch_size), total=len(image_files), desc="Processing images"))
    return results

def display_features_and_numerical_outputs(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image at {image_path}")
        return

    features = extract_features(image)

    # Display extracted features
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Feature Extraction Results')

    axs[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs[0, 0].set_title('Original Image')
    axs[0, 0].axis('off')

    axs[0, 1].imshow(features['gray'], cmap='gray')
    axs[0, 1].set_title('Grayscale')
    axs[0, 1].axis('off')

    axs[0, 2].bar(['Blue', 'Green', 'Red'], features['mean_channels'])
    axs[0, 2].set_title('Mean Channel Values')

    axs[1, 0].imshow(features['edges'], cmap='gray')
    axs[1, 0].set_title('Edge Features')
    axs[1, 0].axis('off')

    axs[1, 1].imshow(features['hog_image'], cmap='gray')
    axs[1, 1].set_title('HOG Features')
    axs[1, 1].axis('off')

    axs[1, 2].axis('off')

    plt.tight_layout()
    plt.show()

    # Print numerical outputs
    print("Features Extracted:")
    print(f"Mean Channel Values: {features['mean_channels']}")
    print(f"Number of Edge Pixels: {np.sum(features['edges'] > 0)}")
    print(f"HOG Features Shape: {features['hog_features'].shape}")
    print(f"HOG Features Mean: {np.mean(features['hog_features'])}")
    print(f"HOG Features Standard Deviation: {np.std(features['hog_features'])}")

if __name__ == "__main__":
    limit = 100  # Set the limit as desired

    # Validation directory paths
    validation_directory = '/content/Pre-Processed_Images/Preprocessed_Images_Validation'
    validation_output_directory = '/content/Feature_Extracted_Images/validation'

    # Process validation images
    validation_image_files = get_image_files(validation_directory, limit=limit)
    if validation_image_files:
        print(f"\nProcessing {len(validation_image_files)} validation images...")
        process_images_batch(validation_image_files, validation_output_directory)
        print("Validation images feature extraction and saving completed.")

        # Display features for a random validation image
        sample_validation_image = random.choice(validation_image_files)
        print(f"\nDisplaying features for randomly selected validation image: {sample_validation_image}")
        display_features_and_numerical_outputs(sample_validation_image)
    else:
        print("No images found in the validation directory.")

# Model Training
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Data Preparation
# Assuming you have your dataset organized in directories for training and validation
train_dir = '/content/Pre-Processed_Images/Preprocessed_Images_Train'
val_dir = '/content/Pre-Processed_Images/Preprocessed_Images_Validation'

# Image data generator for data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

# Step 2: Build the CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(train_generator.class_indices), activation='softmax')  # Number of classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
# Step 3: Train the Model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    epochs=20
)

# Prediction
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import widgets
from IPython.display import display

# Load the model
model = load_model('/content/PlantAI.h5')

# Define a function to preprocess the uploaded image
def preprocess_image(img_path, target_size=(150, 150)):  # Set to (150, 150)
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize pixel values (if necessary)
    return img_array

# Define class names based on the list you provided
class_names = [
    "Apple black rot", "Apple cedar apple rust", "Apple healthy", "Apple scab",
    "Cassava Bacterial Blight (CBB)", "Cassava Brown Streak Disease (CBSD)",
    "Cassava Green Mottle (CGM)", "Cassava Healthy", "Cassava Mosaic Disease (CMD)",
    "Cherry healthy", "Cherry powdery mildew", "Corn cercospora leaf spot gray leaf spot",
    "Corn common rust", "Corn healthy", "Corn northern leaf blight",
    "Grape black rot", "Grape esca (black measles)", "Grape healthy",
    "Grape leaf blight (isariopsis leaf spot)", "Peach bacterial spot", "Peach healthy",
    "Potato early blight", "Potato healthy", "Potato late blight",
    "Rice BrownSpot", "Rice Healthy", "Rice Hispa", "Rice LeafBlast",
    "Strawberry healthy", "Strawberry leaf scorch", "Tomato bacterial spot",
    "Tomato early blight", "Tomato healthy", "Tomato late blight",
    "Tomato leaf mold", "Tomato mosaic virus", "Tomato septoria leaf spot",
    "Tomato spider mites two-spotted spider mite", "Tomato target spot",
    "Tomato yellow leaf curl virus"
]

# Create an upload button widget
upload_button = widgets.FileUpload(accept='image/*', multiple=False)

# Display the upload button
display(upload_button)

# Define a function to handle the uploaded file and make predictions
def on_image_uploaded(change):
    # Get the uploaded file
    uploaded_file = next(iter(upload_button.value.values()))
    content = uploaded_file['content']

    # Save the image to a temporary path
    img_path = "/tmp/uploaded_image.jpg"
    with open(img_path, "wb") as f:
        f.write(content)

    # Preprocess the image
    img_array = preprocess_image(img_path)

    # Make predictions
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    predicted_label = class_names[predicted_class]

    # Display the prediction and image
    img = image.load_img(img_path)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Prediction: {predicted_label}")
    plt.show()
    print(f"Predicted disease: {predicted_label}")

# Link the function to the upload widget
upload_button.observe(on_image_uploaded, names='value')