import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os

# Check if GPU is available
if tf.config.list_physical_devices('GPU'):
    print("GPU is available. Using GPU for training.")
else:
    print("GPU is not available. Training will use CPU.")

# Constants
img_size = 224
batch_size = 32
num_classes = 8  # Adjust based on your dataset
train_data_dir = r"C:\Users\infor\Downloads\BiteSense-Data\train"  # Adjust as needed
test_data_dir = r"C:\Users\infor\Downloads\BiteSense-Data\test"
model_save_path = "final_bug_bite_cnn_model.keras"

# Data augmentation and normalization
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.2  # 20% of data for validation
)

# Load training and validation data
train_data = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="categorical",
    subset="training"
)

val_data = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation"
)

# Load test data without augmentation
test_datagen = ImageDataGenerator(rescale=1.0/255)
test_data = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False  # Keep shuffle=False for consistent testing
)

# Load ResNet50 pre-trained on ImageNet, without the top layers
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3))

# Build new top layers for classification
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Global average pooling layer
x = Dense(1024, activation="relu")(x)  # Dense layer with 1024 units
predictions = Dense(num_classes, activation="softmax")(x)  # Output layer for multi-class classification

# Create the model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze ResNet50 layers during initial training
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])

# Show model summary
model.summary()

# Calculate steps_per_epoch and validation_steps
steps_per_epoch = train_data.samples // batch_size
validation_steps = val_data.samples // batch_size

# Callbacks for checkpointing, early stopping, and learning rate reduction
checkpoint = ModelCheckpoint("best_model.keras", monitor="val_accuracy", save_best_only=True, mode="max")
early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1)

# Train the model with initial frozen layers
history = model.fit(
    train_data,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_data,
    validation_steps=validation_steps,
    epochs=50,  # Start with fewer epochs for frozen training, fine-tune later
    callbacks=[checkpoint, reduce_lr, early_stop]
)

# Unfreeze the last few layers for fine-tuning
for layer in base_model.layers[-10:]:
    layer.trainable = True

# Recompile the model for fine-tuning
model.compile(optimizer=Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])

# Fine-tune the model
history_fine = model.fit(
    train_data,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_data,
    validation_steps=validation_steps,
    epochs=50,  # Fine-tune for additional epochs
    callbacks=[checkpoint, reduce_lr, early_stop]
)

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(test_data)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# Save the final model
model.save(model_save_path)
print(f"Model saved at {model_save_path}")

# Plotting training history
import matplotlib.pyplot as plt

def plot_history(history, title):
    # Plot accuracy and loss
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(len(acc))

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title(f'{title} - Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title(f'{title} - Training and Validation Loss')
    plt.show()

# Plot initial training history
plot_history(history, "Initial Training")

# Plot fine-tuning history
plot_history(history_fine, "Fine-tuning")
