import os
import pickle
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import time


# Load and preprocess images
def load_images(folder, label):
    images = []
    labels = []
    start_time = time.time()
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            end_time = time.time()
            classification_time = end_time - start_time
            print(f"Classification time for {filename}: {classification_time:.2f}s")
            img = cv2.resize(img, (64, 64)).flatten()
            images.append(img)
            labels.append(label)
        else:
            print(f"Failed to load image: {filename}")
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time for loading and preprocessing images: {total_time:.2f}s")
    return images, labels


# Define the paths to the cat and dog image directories and load data sets
cat_dir = '/Users/sterreterpstra/Documents/AI_year_5/Thesis/Dataset/archive/data_set/data_set/cats'
dog_dir = '/Users/sterreterpstra/Documents/AI_year_5/Thesis/Dataset/archive/data_set/data_set/dogs'
cats, labels_cats = load_images(cat_dir, 0)
dogs, labels_dogs = load_images(dog_dir, 1)

# Combine and split the dataset
X = np.array(cats + dogs)
y = np.array(labels_cats + labels_dogs)

# Split data into training, testing and evaluating, 70-15-15
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_test, X_eval, y_test, y_eval = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Train the SVM model
model = SVC(kernel='linear')
start_time = time.time()
model.fit(X_train, y_train)
end_time = time.time()
training_time = end_time - start_time
print(f"Total training time: {training_time:.2f}s")

average_training_time_per_image = training_time / len(X_train)
print(f"Average training time per image: {average_training_time_per_image:.4f}s")

def save_model(model, filename='trained_model.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filename}")

#Evaluate the model for the testing and evaluation sets
y_pred_eval = model.predict(X_eval)
accuracy_eval = accuracy_score(y_eval, y_pred_eval)
print(f"Evaluation Accuracy: {accuracy_eval}")
f1_eval = f1_score(y_eval, y_pred_eval)
print(f"Evaluation F1 Score: {f1_eval}")

y_pred_test = model.predict(X_test)
accuracy_test = accuracy_score(y_test, y_pred_test)
print(f"Test Accuracy: {accuracy_test}")
f1_test = f1_score(y_test, y_pred_test)
print(f"Test F1 Score: {f1_test}")

# Save the trained model
save_model(model)