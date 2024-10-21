import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load your model
model = tf.keras.models.load_model(r"C:\Users\Saaki\Downloads\best_model~0.8125.keras")

# Function to load and preprocess the image
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Resize image
    img_array = image.img_to_array(img)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize to [0, 1] range
    return img_array

# Function to predict the class of the image and get probabilities
def predict_image_class(img_path):
    img_array = load_and_preprocess_image(img_path)
    predictions = model.predict(img_array)

    # Get predicted class index and probabilities
    predicted_class_index = np.argmax(predictions, axis=1)[0]  # Get index of the highest score
    predicted_probabilities = predictions[0]  # Get probabilities for each class

    # Map the predicted class index back to class labels
    class_labels = ["Ant", "Bedbug", "Chigger", "Flea", "Mosquito", "No_Bite", "Spider", "Tick"]
    predicted_class = class_labels[predicted_class_index]

    return predicted_class, predicted_probabilities

# Test the model with your image
img_path = r"C:\Users\Saaki\Downloads\BiteSense-Data\train\Bedbug\bed_bugsimage49.jpg"  # Replace with path
predicted_class, predicted_probabilities = predict_image_class(img_path)

# Display the results
print(f"The predicted class for the image is: {predicted_class}")
print("Probabilities for each class:")
for i, class_name in enumerate(["Ant", "Bedbug", "Chigger", "Flea", "Mosquito", "No_Bite", "Spider", "Tick"]):
    print(f"{class_name}: {predicted_probabilities[i]:.4f}")
