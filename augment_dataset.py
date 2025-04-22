
import os
import shutil
import numpy as np
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img

# Set paths
input_dir = "/content/drive/MyDrive/fyproject/breakHis/Breakhis_400x/"  # Update with your input folder path
output_dir = "/content/drive/MyDrive/fyproject/breakhis_aug/Breakhis_400x/"  # Update with your desired output folder path

# Create output directories
os.makedirs(output_dir, exist_ok=True)
for cls in ['Benign', 'Malignant']:
    os.makedirs(os.path.join(output_dir, cls), exist_ok=True)

# Load dataset
benign_dir = os.path.join(input_dir, "Benign")
malignant_dir = os.path.join(input_dir, "Malignant")

benign_images = os.listdir(benign_dir)
malignant_images = os.listdir(malignant_dir)

# Balance dataset by augmenting the minority class
benign_count = len(benign_images)
malignant_count = len(malignant_images)

target_count = max(benign_count, malignant_count)  # Balance to the larger class

def augment_and_save(image_path, save_dir, augment_count):
    img = load_img(image_path)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    i = 0
    for batch in datagen.flow(img_array, batch_size=1, save_to_dir=save_dir, save_prefix="aug", save_format="jpeg"):
        i += 1
        if i >= augment_count:
            break

# Augment and save images to balance classes
for cls, img_list, cls_dir in zip(['Benign', 'Malignant'], [benign_images, malignant_images], [benign_dir, malignant_dir]):
    current_count = len(img_list)
    save_dir = os.path.join(output_dir, cls)

    for img_name in img_list:
        img_path = os.path.join(cls_dir, img_name)
        shutil.copy(img_path, save_dir)  # Copy original images

    additional_needed = target_count - current_count
    if additional_needed > 0:
        random.shuffle(img_list)
        for i in range(additional_needed):
            augment_and_save(os.path.join(cls_dir, img_list[i % current_count]), save_dir, augment_count=1)

# Print final class counts
final_benign_count = len(os.listdir(os.path.join(output_dir, "Benign")))
final_malignant_count = len(os.listdir(os.path.join(output_dir, "Malignant")))

print("Augmented dataset saved to:", output_dir)
print(f"Final dataset size: Benign - {final_benign_count}, Malignant - {final_malignant_count}")
