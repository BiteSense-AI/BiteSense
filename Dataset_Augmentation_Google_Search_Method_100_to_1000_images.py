import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array


original_images_path = os.path.join(os.path.expanduser("~"), "Downloads", "mosquito_bite_images")
augmented_images_path = os.path.join(os.path.expanduser("~"), "Downloads", "augmented_mosquito_bite_images")
os.makedirs(augmented_images_path, exist_ok=True)

datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

image_files = [f for f in os.listdir(original_images_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
total_images_needed = 1000
current_image_count = len(image_files)
images_per_original = (total_images_needed // current_image_count) + 1


for image_file in image_files:
    img = load_img(os.path.join(original_images_path, image_file))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    for i, batch in enumerate(datagen.flow(img_array, batch_size=1, save_to_dir=augmented_images_path, save_prefix='aug_', save_format='jpeg')):
        if i >= images_per_original:
            break

print(f"Augmented images saved to {augmented_images_path}")
