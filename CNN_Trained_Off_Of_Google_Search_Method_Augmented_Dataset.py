import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split

augmented_images_path = os.path.join(os.path.expanduser("~"), "Downloads", "augmented_mosquito_bite_images")

def load_data(data_dir):
    images = []
    labels = []

    for img_file in os.listdir(data_dir):
        if img_file.endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(data_dir, img_file)
            img = load_img(img_path, target_size=(128, 128))
            img_array = img_to_array(img)
            images.append(img_array)
            labels.append(1)

    return np.array(images), np.array(labels)

X, y = load_data(augmented_images_path)
X = X / 255.0
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the CNN
def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="sigmoid"))
    return model

model = create_model()
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=3, batch_size=32, validation_data=(X_val, y_val))
model.save("mosquito_bite_classifier.keras")

def predict_mosquito_bite(image_path):
    img = load_img(image_path, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    probability = prediction[0][0]
    print(f"Probability of being a mosquito bite: {probability * 100:.2f}%")
    return probability