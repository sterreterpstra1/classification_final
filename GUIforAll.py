from random import random, choice
import numpy as np
import os
import tkinter as tk
from tensorflow import keras
from tkinter import filedialog
from PIL import ImageTk, Image

def load_CNN():
    # Specify the path to the saved model file, including the filename
    model_path = '/Users/sterreterpstra/PycharmProjects/pythonProject4/cat_dog_classifier_1.h5'
    # Load the saved model
    model_s = keras.models.load_model(model_path)
    return model_s


def load_CNNbiased():
    # Specify the path to the saved model file, including the filename
    model_path = '/Users/sterreterpstra/PycharmProjects/pythonProject4/cat_dog_classifierbiased_1.h5'
    # Load the saved model
    model_s = keras.models.load_model(model_path)
    return model_s


def load_DT():
    model_path = '/Users/sterreterpstra/PycharmProjects/pythonProject4/cat_dog_DT.h5'
    model_s = keras.models.load_model(model_path)
    return model_s


def load_SVM():
    model_path = '/Users/sterreterpstra/PycharmProjects/pythonProject4/cat_dog_SVM.h5'
    model_s = keras.models.load_model(model_path)
    return model_s


# Load the trained model
model_saved_CNN = load_CNN()
model_saved_CNNbiased = load_CNNbiased()
model_saved_DT = load_DT()
model_saved_SVM = load_SVM()


def upload_image_CNN():
    global image_label, predicted_category_label, predicted_images_frame
    # Open the file dialog to select an image
    filename = filedialog.askopenfilename(initialdir='.', title='Select Image',
                                          filetypes=[('Image Files', '*.jpg .jpeg .png')])
    # Check if an image was selected
    if filename:
        image = Image.open(filename)  # Load the image from the selected file
        image = image.resize((64, 64))  # Resize the image to 64x64
        image = image.convert('RGB')  # Convert the image to RGB format
        image_array = np.array(image)  # Convert the image to NumPy array for prediction
        image_array = image_array.reshape((1, 64, 64, 3))  # Reshape the image array to fit the model input shape

        prediction = model_saved_CNN.predict(image_array)
        predicted_category = int(prediction[0])  # Update this line to match your model's output
        predicted_category_label_text = 'Cat' if predicted_category == 0 else 'Dog'  # Convert the binary prediction to a human-readable label
        predicted_category_label.config(
            text=f'Predicted category: {predicted_category_label_text}')  # Update the GUI with the predicted category
        predicted_category_label.update()

        # Display the selected image
        image_label.destroy()  # Clear the existing image label

        # Convert the Image object to an ImageTk object
        image_tk = ImageTk.PhotoImage(image)

        # Create a label to display the ImageTk object
        image_label = tk.Label(image_frame, image=image_tk)
        image_label.pack()

        predicted_images_frame.destroy()  # Clear the existing predicted images frame
        predicted_images_frame = tk.Frame(root)
        predicted_images_frame.pack()

def upload_image_CNNbiased():
    global image_label, predicted_category_label, predicted_images_frame
    # Open the file dialog to select an image
    filename = filedialog.askopenfilename(initialdir='.', title='Select Image',
                                          filetypes=[('Image Files', '*.jpg .jpeg .png')])
    # Check if an image was selected
    if filename:
        image = Image.open(filename)  # Load the image from the selected file
        image = image.resize((64, 64))  # Resize the image to 64x64
        image = image.convert('RGB')  # Convert the image to RGB format
        image_array = np.array(image)  # Convert the image to NumPy array for prediction
        image_array = image_array.reshape((1, 64, 64, 3))  # Reshape the image array to fit the model input shape

        prediction = model_saved_CNN.predict(image_array)
        predicted_category = int(prediction[0])  # Update this line to match your model's output
        predicted_category_label_text = 'Cat' if predicted_category == 0 else 'Dog'  # Convert the binary prediction to a human-readable label
        predicted_category_label.config(
            text=f'Predicted category: {predicted_category_label_text}')  # Update the GUI with the predicted category
        predicted_category_label.update()

        # Display the selected image
        image_label.destroy()  # Clear the existing image label

        # Convert the Image object to an ImageTk object
        image_tk = ImageTk.PhotoImage(image)

        # Create a label to display the ImageTk object
        image_label = tk.Label(image_frame, image=image_tk)
        image_label.pack()

        predicted_images_frame.destroy()  # Clear the existing predicted images frame
        predicted_images_frame = tk.Frame(root)
        predicted_images_frame.pack()


def upload_image_DT():
    global image_label, predicted_category_label, predicted_images_frame
    # Open the file dialog to select an image
    filename = filedialog.askopenfilename(initialdir='.', title='Select Image',
                                          filetypes=[('Image Files', '*.jpg .jpeg .png')])
    # Check if an image was selected
    if filename:
        image = Image.open(filename)  # Load the image from the selected file
        image = image.resize((64, 64))  # Resize the image to 64x64
        image = image.convert('RGB')  # Convert the image to RGB format
        image_array = np.array(image)  # Convert the image to NumPy array for prediction
        image_array = image_array.reshape((1, 64, 64, 3))  # Reshape the image array to fit the model input shape

        prediction = model_saved_SVM.predict(image_array)
        predicted_category = int(prediction[0])  # Update this line to match your model's output
        predicted_category_label_text = 'Cat' if predicted_category == 0 else 'Dog'  # Convert the binary prediction to a human-readable label
        predicted_category_label.config(
            text=f'Predicted category: {predicted_category_label_text}')  # Update the GUI with the predicted category
        predicted_category_label.update()

        # Display the selected image
        image_label.destroy()  # Clear the existing image label

        # Convert the Image object to an ImageTk object
        image_tk = ImageTk.PhotoImage(image)

        # Create a label to display the ImageTk object
        image_label = tk.Label(image_frame, image=image_tk)
        image_label.pack()

        predicted_images_frame.destroy()  # Clear the existing predicted images frame
        predicted_images_frame = tk.Frame(root)
        predicted_images_frame.pack()

def upload_image_SVM():
    global image_label, predicted_category_label, predicted_images_frame
    # Open the file dialog to select an image
    filename = filedialog.askopenfilename(initialdir='.', title='Select Image',
                                          filetypes=[('Image Files', '*.jpg .jpeg .png')])
    # Check if an image was selected
    if filename:
        image = Image.open(filename)  # Load the image from the selected file
        image = image.resize((64, 64))  # Resize the image to 64x64
        image = image.convert('RGB')  # Convert the image to RGB format
        image_array = np.array(image)  # Convert the image to NumPy array for prediction
        image_array = image_array.reshape((1, 64, 64, 3))  # Reshape the image array to fit the model input shape

        prediction = model_saved_DT.predict(image_array)
        predicted_category = int(prediction[0])  # Update this line to match your model's output
        predicted_category_label_text = 'Cat' if predicted_category == 0 else 'Dog'  # Convert the binary prediction to a human-readable label
        predicted_category_label.config(
            text=f'Predicted category: {predicted_category_label_text}')  # Update the GUI with the predicted category
        predicted_category_label.update()

        # Display the selected image
        image_label.destroy()  # Clear the existing image label

        # Convert the Image object to an ImageTk object
        image_tk = ImageTk.PhotoImage(image)

        # Create a label to display the ImageTk object
        image_label = tk.Label(image_frame, image=image_tk)
        image_label.pack()

        predicted_images_frame.destroy()  # Clear the existing predicted images frame
        predicted_images_frame = tk.Frame(root)
        predicted_images_frame.pack()


# Create the GUI
root = tk.Tk()
root.title('Cat vs Dog Classifier')
root.geometry('800x600')

# Create a frame for the image upload
upload_frame = tk.Frame(root)
upload_frame.pack()

# Create a button for the CNN model
cnn_button = tk.Button(upload_frame, text='Convolutional Neural Network', command=upload_image_CNN)
cnn_button.pack()

cnnbiased_button = tk.Button(upload_frame, text='Convolutional Neural Network 2', command=upload_image_CNNbiased)
cnnbiased_button.pack()

# Create a button for the SVM model
svm_button = tk.Button(upload_frame, text='Support Vector Machine', command=upload_image_SVM)
svm_button.pack()

# Create a button for the DT model
dt_button = tk.Button(upload_frame, text='Decision Tree', command=upload_image_DT)
dt_button.pack()

# Create a frame for the image display
image_frame = tk.Frame(root)
image_frame.pack()

# Create a label to display the uploaded image
image_label = tk.Label(image_frame)
image_label.pack()

# Create a frame for the prediction results
results_frame = tk.Frame(root)
results_frame.pack()

# Create a label to display the predicted category
predicted_category_label = tk.Label(results_frame, text='Predicted category:')
predicted_category_label.pack()

# Create a frame to display the predicted images
predicted_images_frame = tk.Frame(root)
predicted_images_frame.pack()

# Start the GUI
root.mainloop()
