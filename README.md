# Image Classification Model with TensorFlow

This project implements an image classification model using **TensorFlow** and **Keras** in Python. The model is a **Convolutional Neural Network (CNN)** designed to classify images into categories, with training, validation, and testing phases. Below are the details of the model, environment setup, and execution steps specifically for a **Mac** environment.

## Table of Contents

- [Environment Setup (Mac)](#environment-setup-mac)
- [Model Overview](#model-overview)
- [Training and Validation Performance](#training-and-validation-performance)
- [Code Structure](#code-structure)
- [Execution](#execution)

---

## Environment Setup (Mac)

To set up the environment and run this project on a Mac, follow these steps:

1. Create a Python Virtual Environment: `python -m venv imageclassification`

2. Activate the Python Virtual Environment: `source imageclassification/bin/activate`

3. Install Jupyter Notebook Kernel: `pip install ipykernel`

4. Associate the Environment with Jupyter Notebook: `python -m ipykernel install --name=imageclassification`

5. Install Required Libraries: Install TensorFlow, OpenCV, and Matplotlib for building and visualizing the model. `pip install tensorflow opencv-python matplotlib`

6. Launch Jupyter Notebook: You can launch Jupyter Notebook with the following command: `/Applications/anaconda3/bin/jupyter_mac.command`

---

## Model Overview

This model is a **Convolutional Neural Network (CNN)** designed for image classification tasks. The architecture consists of multiple convolutional layers, max pooling, fully connected layers, and dropout for regularization. The model has been compiled using the **Adam optimizer** with **binary cross-entropy loss**, ideal for binary classification problems.

### Model Architecture:

1. **Conv2D Layers**: Extract features from the image.
2. **MaxPooling2D**: Downsample the feature maps.
3. **Dense Layers**: Fully connected layers for classification.
4. **Sigmoid Output**: For binary classification.

---

## Training and Validation Performance

### Accuracy over Epochs

The graph below shows the training and validation accuracy of the model over 20 epochs.

![Model Accuracy](images/accuracy_plot.png)

### Loss over Epochs

The graph below shows the training and validation loss of the model over 20 epochs.

![Model Loss](images/loss_plot.png)

---

## Code Structure

Below is a quick overview of the code structure and the key components of the model:

### Loading and Preprocessing Data

The image dataset is loaded using TensorFlow's utility and is preprocessed by normalizing the pixel values (scaling between 0 and 1).

`data = tf.keras.utils.image_dataset_from_directory('data')`  
`data = data.map(lambda x, y: (x / 255, y))`

### Model Definition

The model is built using the **Sequential API** of Keras, stacking layers sequentially to create a feed-forward neural network. Below is the architecture.

`model = Sequential()`  
`model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))`  
`model.add(MaxPooling2D())`  
`model.add(Conv2D(32, (3,3), 1, activation='relu'))`  
`model.add(MaxPooling2D())`  
`model.add(Conv2D(16, (3,3), 1, activation='relu'))`  
`model.add(MaxPooling2D())`  
`model.add(Flatten())`  
`model.add(Dense(256, activation='relu'))`  
`model.add(Dense(1, activation='sigmoid'))`

### Compiling the Model

The model is compiled using the Adam optimizer and binary cross-entropy as the loss function, suitable for binary classification.

`model.compile(optimizer='adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])`

### Training the Model

The model is trained over 20 epochs with validation data using the **TensorBoard** callback to monitor the metrics.

`logdir='logs'`  
`tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)`  
`hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])`

### Plotting Accuracy and Loss

Here are the visualizations of accuracy and loss for both training and validation sets:

`fig, ax = plt.subplots(figsize=(10, 6))`  
`ax.plot(hist.history['accuracy'], color='teal', label='Training Accuracy')`  
`ax.plot(hist.history['val_accuracy'], color='orange', label='Validation Accuracy')`  
`ax.set_title('Model Accuracy Over Epochs')`  
`ax.set_xlabel('Epochs')`  
`ax.set_ylabel('Accuracy')`  
`ax.legend(loc='lower right')`  
`ax.grid(True)`  
`plt.show()`

`fig, ax = plt.subplots(figsize=(10, 6))`  
`ax.plot(hist.history['loss'], color='blue', label='Training Loss')`  
`ax.plot(hist.history['val_loss'], color='red', label='Validation Loss')`  
`ax.set_title('Model Loss Over Epochs')`  
`ax.set_xlabel('Epochs')`  
`ax.set_ylabel('Loss')`  
`ax.legend(loc='upper right')`  
`ax.grid(True)`  
`plt.show()`

---

## Execution

To run this project, follow the steps below:

1. Clone the Repository: `git clone <repository-url>`

2. Navigate to the Project Directory: `cd imageclassification`

3. Set Up the Environment: Create and activate the Python environment as outlined in the setup steps.

4. Launch Jupyter Notebook: `/Applications/anaconda3/bin/jupyter_mac.command`

5. Run the Notebook: Open the notebook file (`main.ipynb`) and execute the cells to train the model, evaluate performance, and view the accuracy/loss plots.

---

## TensorBoard Visualization

After running the model training, you can launch TensorBoard to visualize the logs:

`tensorboard --logdir=logs`

Navigate to the provided URL to view training and validation metrics in real time.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
