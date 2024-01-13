import os
import time
import numpy as np
from skimage import io, transform, color
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from joblib import dump, load
import h5py

# Define the paths to the cat and dog image directories
cat_dir = '/Users/sterreterpstra/Documents/AI_year_5/Thesis/Dataset/archive/data_set/data_set/cats'
dog_dir = '/Users/sterreterpstra/Documents/AI_year_5/Thesis/Dataset/archive/data_set/data_set/dogs'

# Load and preprocess images
def load_images_from_directory(image_dir, label):
    image_list = []
    labels = []
    for filename in os.listdir(image_dir):
        if filename.endswith(".jpg"):
            img = io.imread(os.path.join(image_dir, filename))
            img = transform.resize(img, (64, 64), mode='reflect')
            img = color.rgb2gray(img)
            image_list.append(img)
            labels.append(label)
    return image_list, labels

cat_images, cat_labels = load_images_from_directory(cat_dir, 0)
dog_images, dog_labels = load_images_from_directory(dog_dir, 1)

# Combine cat and dog images and labels
X = cat_images + dog_images
y = cat_labels + dog_labels

# Convert data to numpy arrays
X = np.array(X)
y = np.array(y)

# Flatten each image to a 1D array
X = X.reshape(len(X), -1)

# Split the data into training and testing sets
start_time = time.time()
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_test, X_eval, y_test, y_eval = train_test_split(X_temp, y_temp, test_size=1 / 3, random_state=42)
end_training_time = time.time()
training_time = end_training_time - start_time
print("Training time for one image:", training_time)

# Create and train the Decision Tree classifier
decision_tree_classifier = DecisionTreeClassifier()
decision_tree_classifier.fit(X_train, y_train)

# Save the model
serialized_model = dump(decision_tree_classifier, 'cat_dog_DT.joblib')
h5f = h5py.File('cat_dog_DT.h5', 'w')
h5f['model'] = serialized_model
h5f.close()

# Make predictions on the test data
y_pred = decision_tree_classifier.predict(X_test)

# Evaluate the model
evaluation_accuracy = accuracy_score(y_eval, decision_tree_classifier.predict(X_eval))
evaluation_f1_score = f1_score(y_eval, decision_tree_classifier.predict(X_eval))
print("Evaluation Accuracy: {:.2f}%".format(evaluation_accuracy * 100))
print("Evaluation F1-Score: {:.2f}".format(evaluation_f1_score))
