from random import random, choice
import numpy as np
import os
import tkinter as tk
from tkinter import ttk
# from tensorflow import keras TODO: Enable this line when the model is loaded
from tkinter import filedialog
from PIL import ImageTk, Image
import random
from PIL import Image

# Load the saved model
def load_model(model_key):
    # Specify the path to the saved model file, including the filename
    model_path = {'svm': 'classifier_SVM.h5', 
                  'dt': 'classifier_DT.h5', 
                  'cnn': 'classifier_CNN.h5', 
                  'cnn_bias': 'classifier_CNN_biased.h5'
                  }

    # Load the saved model
    # model_s = keras.models.load_model(model_path[model_key]) TODO: Enable this line when the model is loaded
    model_s = None
    return model_s

def upload_image():
    global image_label, image_array 

    # Open the file dialog to select an image
    filename = filedialog.askopenfilename(initialdir='.', title='Select Image',
                                          filetypes=[('Image Files', '*.jpg .jpeg .png')])
    
    # Load image
    try:
        image_raw = Image.open(filename)  # Load the image from the selected file
        
    except Exception as e:
        print(f"Error loading image: {e}")

    # Image preprocessing
    image = image_raw.resize((64, 64))  # Resize the image to 64x64
    image = image.convert('RGB')  # Convert the image to RGB format
    image_array = np.array(image)  # Convert the image to NumPy array for prediction
    image_array = image_array.reshape((1, 64, 64, 3))  # Reshape the image array to fit the model input shape
    
    # Clean the existing image_label
    image_label.destroy()
    
    # Create an object to show default-img.jpg
    image_object = ImageTk.PhotoImage(image_raw.resize((300, 300)))

    # Create a label to display the default image
    image_label = tk.Label(image_frame, image=image_object)
    image_label.pack(side='left', padx=5, pady=5)
    
    # Display the image in the frame
    image_label.image = image_object
    
    return image_array

def predict_category(model):
    global image_label, predicted_category_label, image_array

    # Predict the category
    prediction = random.randint(0, 1)
    # prediction = model.predict(image_array) #TODO: Enable this line when the model is loaded
    
    # Convert the prediction to a human-readable label
    predicted_category = int(prediction)  # TODO: Update this line to match your model's output
    predicted_category_label_text = 'Cat' if predicted_category == 0 else 'Dog'  # Convert the binary prediction to a human-readable label
    predicted_category_label.config(
        text=f'Predicted category: {predicted_category_label_text}')  # Update the GUI with the predicted category
    predicted_category_label.update()

    return predicted_category

def update_similar_images(predicted_category):
    global image_similar_object_1, image_similar_object_2, image_similar_object_3, similar_label_1, similar_label_2, similar_label_3

    # Get the list of images in the predicted category
    if predicted_category == 0:
        image_list = os.listdir('images/cats')
    else:
        image_list = os.listdir('images/dogs')

    # Get 3 random images from the list
    random_images = random.sample(image_list, 3)

    # Update the image objects
    image_similar_object_1 = ImageTk.PhotoImage(Image.open(os.path.join('images/cats' if predicted_category == 0 else 'images/dogs', random_images[0])).resize((300, 300)))
    image_similar_object_2 = ImageTk.PhotoImage(Image.open(os.path.join('images/cats' if predicted_category == 0 else 'images/dogs', random_images[1])).resize((300, 300)))
    image_similar_object_3 = ImageTk.PhotoImage(Image.open(os.path.join('images/cats' if predicted_category == 0 else 'images/dogs', random_images[2])).resize((300, 300)))

    # Update the labels
    similar_label_1.config(image=image_similar_object_1)
    similar_label_1.image = image_similar_object_1
    similar_label_1.update()

    similar_label_2.config(image=image_similar_object_2)
    similar_label_2.image = image_similar_object_2
    similar_label_2.update()

    similar_label_3.config(image=image_similar_object_3)
    similar_label_3.image = image_similar_object_3
    similar_label_3.update()

def main_button(model_key):
    global image

    # Load the model
    model = load_model(model_key)

    # Predict the category
    predicted_category = predict_category(model)

    update_similar_images(predicted_category)
    
####################
# Create the GUI
root = tk.Tk()
root.title('Cat vs Dog Classifier')
root.geometry('1000x950')

# Create a frame for the title
title_frame = tk.Frame(root)
title_frame.pack()

# Create a label for the title
title_label = tk.Label(title_frame, text='Cat vs Dog Classifier', font=('Arial', 20, 'bold'))
title_label.pack(pady=10)

# Create a frame for the image upload
image_frame = tk.Frame(root)
image_frame.pack()

# Create an object to show default-img.jpg
image_object = ImageTk.PhotoImage(Image.open("default-img.jpg"))

# Create a label to display the default image
image_label = tk.Label(image_frame, image=image_object)
image_label.pack(side='left', padx=5, pady=5)

# Create a frame for the image upload button
upload_frame = tk.Frame(root)
upload_frame.pack()

# Create a button to upload the image
upload_button = tk.Button(upload_frame, 
                          text='Upload Image', 
                          command=lambda: upload_image())
upload_button.pack(side='left', padx=5, pady=5)

# Create a frame for the buttons
model_frame = tk.Frame(root)
model_frame.pack()

# Create a label for the title
title_label = tk.Label(model_frame, text='Select a classifier:', font=('Arial', 16))
title_label.pack(pady=10)

# Create a button for the SVM model
svm_button = tk.Button(model_frame, 
                       text='Support Vector Machine', 
                       command=lambda: main_button(model_key='svm'))
svm_button.pack(side='left', padx=5, pady=5)

# Create a button for the DT model
dt_button = tk.Button(model_frame, 
                      text='Decision Tree', 
                      command=lambda: main_button(model_key='dt'))
dt_button.pack(side='left', padx=5, pady=5)

# Create a button for the biased CNN model
cnn_button = tk.Button(model_frame, 
                       text='Convolutional Neural Network 1', 
                       command=lambda: main_button(model_key='cnn_bias'))
cnn_button.pack(side='left', padx=5, pady=5)

# Create a button for the CNN model
cnnbias_button = tk.Button(model_frame, 
                           text='Convolutional Neural Network 2', 
                           command=lambda: main_button(model_key='cnn'))
cnnbias_button.pack(side='left', padx=5, pady=5)

# Create a frame to display the predicted images
results_frame = tk.Frame(root)
results_frame.pack()

# Create a label to display the predicted category
predicted_category_label = tk.Label(results_frame, text='Predicted category:', font=('Arial', 16))
predicted_category_label.pack(side='left', padx=5, pady=15)

# Create a frame to display the predicted images
similar_img_frame = tk.Frame(root)
similar_img_frame.pack()

# Add a horizontal divider
divider = ttk.Separator(similar_img_frame, orient='horizontal')
divider.pack(fill='x', padx=20, pady=20)

# Create a label to display the predicted category
similar_img_label = tk.Label(similar_img_frame, text='Similar images in the same category', font=('Arial', 16))
similar_img_label.pack()

# Create an object to show default-img.jpg
image_similar_object_1 = ImageTk.PhotoImage(Image.open("default-img.jpg"))
image_similar_object_2 = ImageTk.PhotoImage(Image.open("default-img.jpg"))
image_similar_object_3 = ImageTk.PhotoImage(Image.open("default-img.jpg"))

# Create a label to display the default image
similar_label_1 = tk.Label(similar_img_frame, image=image_similar_object_1)
similar_label_1.pack(side='left', padx=5, pady=5)

similar_label_2 = tk.Label(similar_img_frame, image=image_similar_object_2)
similar_label_2.pack(side='left', padx=5, pady=5)

similar_label_3 = tk.Label(similar_img_frame, image=image_similar_object_3)
similar_label_3.pack(side='left', padx=5, pady=5)

# Start the GUI
root.mainloop()
