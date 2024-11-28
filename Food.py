import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# Set dataset paths
data_dir = r'C:\Users\HP\OneDrive\Desktop\Intern\FooDD'  # Update with your dataset path

# Custom data generator to handle corrupted images
def safe_data_generator(generator):
    while True:
        try:
            yield next(generator)
        except OSError as e:
            print(f"Skipping a corrupted image. Error: {e}")
            continue

# ImageDataGenerator for augmentation and normalization
data_gen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=0.2,  # 80-20 split for training and validation
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

# Training data generator
train_data = safe_data_generator(data_gen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    subset="training"
))

# Validation data generator
val_data = safe_data_generator(data_gen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    subset="validation"
))

# Check the number of classes and generate calorie mapping dynamically
class_indices = data_gen.flow_from_directory(data_dir, subset="training").class_indices
num_classes = len(class_indices)
print(f"Number of classes: {num_classes}")

# Dynamic calorie mapping (update calorie values based on dataset classes)
calorie_mapping = {
    class_name: np.random.randint(50, 400)  # Replace with actual calorie values if known
    for class_name in class_indices.keys()
}
print(f"Calorie Mapping: {calorie_mapping}")

# Load pre-trained EfficientNetB0 as the base model
base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Add custom classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation="softmax")(x)

# Combine base model and custom layers
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the base model layers for initial training
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Callbacks for training
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-6)

# Train the model (initial phase)
history = model.fit(
    train_data,
    validation_data=val_data,
    steps_per_epoch=100,
    validation_steps=50,
    epochs=15,
    callbacks=[early_stop, reduce_lr]
)

# Fine-tune the base model (unfreeze layers)
for layer in base_model.layers:
    layer.trainable = True

# Recompile the model with a lower learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss="categorical_crossentropy", metrics=["accuracy"])

# Fine-tune training
history_finetune = model.fit(
    train_data,
    validation_data=val_data,
    steps_per_epoch=100,
    validation_steps=50,
    epochs=10,
    callbacks=[early_stop, reduce_lr]
)

# Evaluate the model
val_loss, val_acc = model.evaluate(val_data, steps=50)
print(f"Validation Accuracy: {val_acc:.2f}")

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Loss")
plt.legend()
plt.show()

# Prediction function with calorie estimation
def predict_and_estimate(image_path, model, calorie_mapping):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    class_idx = np.argmax(pred)
    class_name = list(calorie_mapping.keys())[class_idx]
    calories = calorie_mapping[class_name]

    print(f"Predicted Food: {class_name}, Estimated Calories: {calories}")

# Test prediction
test_image = "path_to_test_image.jpg"  # Replace with the actual test image path
predict_and_estimate(test_image, model, calorie_mapping)

# Save the trained model
model.save("food_calorie_model.h5")
 