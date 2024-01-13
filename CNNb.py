import os
import numpy as np
from skimage import io, transform, color
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping


# Paths to image directories
cat_dir = '/Users/sterreterpstra/Documents/AI_year_5/Thesis/BiasedDataSet/cats'
dog_dir = '/Users/sterreterpstra/Documents/AI_year_5/Thesis/BiasedDataSet/dogs'

# Load and preprocess images
def load_images_from_directory(image_dir, label):
    image_list = []
    labels = []
    for filename in os.listdir(image_dir):
        if filename.endswith(".jpg"):
            img = io.imread(os.path.join(image_dir, filename))
            img_resized = transform.resize(img, (64, 64), mode='reflect')
            image_list.append(img_resized)
            labels.append(label)
    return image_list, labels


cat_images, cat_labels = load_images_from_directory(cat_dir, 0)
dog_images, dog_labels = load_images_from_directory(dog_dir, 1)

# Combine cat and dog images and labels
X = cat_images + dog_images
y = cat_labels + dog_labels

# Convert data to numpy arrays: .reshape (-1 64 64 1)
X = np.array(X)
y = np.array(y)

# Split into testing, training and evaluating 70-15-15
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_test, X_eval, y_test, y_eval = train_test_split(X_temp, y_temp, test_size=1 / 3, random_state=42)

# Data augmentation for training data
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

datagen.fit(X_train)


model = keras.Sequential([
    layers.Conv2D(32, (3, 3), input_shape=(64, 64, 3)),
    BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    Dropout(0.25),
    layers.Conv2D(64, (3, 3)),
    BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    Dropout(0.25),
    layers.Conv2D(128, (3, 3)),
    BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    Dropout(0.25),
    layers.Conv2D(256, (3, 3)),
    BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128),
    BatchNormalization(),
    layers.Activation('relu'),
    layers.Dense(1, activation='sigmoid')
])

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,         # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True
)


# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=100,
    validation_data=(X_eval, y_eval),
    callbacks=[early_stopping]  # Add the callback here
)

model.save('cat_dog_classifierbiased_1.h5')

# Evaluate the model
test_accuracy = model.evaluate(X_test, y_test, verbose=0)[1]
y_pred_test = (model.predict(X_test) > 0.5).astype(int)

print("Test Set Accuracy: {:.2f}%".format(test_accuracy * 100))
print("Test Set Confusion Matrix:\n", confusion_matrix(y_test, y_pred_test))
print("Test Set Classification Report:\n", classification_report(y_test, y_pred_test))

eval_accuracy = model.evaluate(X_eval, y_eval, verbose=0)[1]
y_pred_eval = (model.predict(X_eval) > 0.5).astype(int)

print("Evaluation Set Accuracy: {:.2f}%".format(eval_accuracy * 100))
print("Evaluation Set Confusion Matrix:\n", confusion_matrix(y_eval, y_pred_eval))
print("Evaluation Set Classification Report:\n", classification_report(y_eval, y_pred_eval))


def plot_loss(history):
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, 'bo-', label='Training Loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
    plt.title('Training, Testing and Validation Loss', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()

plot_loss(history)