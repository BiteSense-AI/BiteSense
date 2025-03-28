import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Set constants for image size and batch size
img_size = 224
batch_size = 32
num_classes = 8  # Reminder to self - Edit based on # classes

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

# Load training data
train_data = train_datagen.flow_from_directory(
    r"C:\Users\infor\Downloads\BiteSense-Data\train",  # Change to proper path
    target_size=(img_size, img_size),  # Resizing images to 224x224
    batch_size=batch_size,
    class_mode="categorical",
    subset="training"
)

# Load validation data
val_data = train_datagen.flow_from_directory(
    r"C:\Users\infor\Downloads\BiteSense-Data\train",  # Same as training
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation"
)

# Load testing data w/o data augmentation
test_datagen = ImageDataGenerator(rescale=1.0/255)

test_data = test_datagen.flow_from_directory(
    r"C:\Users\infor\Downloads\BiteSense-Data\test",  # Path to test dataset
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False  # Keep shuffle=False for testing
)

# Load ResNet50 with pre-trained ImageNet weights, excluding the top layer
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3))

# Add new top layers for classification
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Global average pooling
x = Dense(1024, activation="relu")(x)  # Fully connected layer with 1024 units
predictions = Dense(num_classes, activation="softmax")(x)  # Output layer with softmax for multi-class classification

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers of ResNet50
for layer in base_model.layers:
    layer.trainable = False

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])

# Calculate steps_per_epoch and validation_steps
steps_per_epoch = train_data.samples // batch_size
validation_steps = val_data.samples // batch_size

# Set up checkpoints, early stopping, and learning rate adjustment
checkpoint = ModelCheckpoint("best_model.keras", monitor="val_accuracy", save_best_only=True, mode="max")
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1)

# Train the model
history = model.fit(
    train_data,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_data,
    validation_steps=validation_steps,
    epochs=50,
    callbacks=[checkpoint, reduce_lr]
)

# Unfreeze the last few layers of the base model for fine-tuning
for layer in base_model.layers[-10:]:
    layer.trainable = True

# Recompile the model after unfreezing layers
model.compile(optimizer=Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])


# Evaluate model on test data
test_loss, test_accuracy = model.evaluate(test_data)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# Save model
model.save("final_bug_bite_cnn_model.keras")
