import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the trained model
model = load_model('mosquito_bite_classifier.keras')

# Function to if mosquito bite
def predict_mosquito_bite(image_path):
    img = load_img(image_path, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    probability = prediction[0][0]
    print(f"Probability of being a mosquito bite: {probability * 100:.2f}%")
    return probability

predict_mosquito_bite(r"C:\Users\Saaki\Downloads\dog.jpg")
