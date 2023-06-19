import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in {2, 3}:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=TEST_SIZE)

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Loads image data from a directory and returns a NumPy array of images and corresponding labels.

    Args:
        data_dir (str): The directory path where the image data is stored. The directory should have subdirectories
                        named after the labels, and each subdirectory should contain the corresponding images.

    Returns:
        images (numpy.ndarray): A NumPy array containing the loaded images. The shape of the array is (num_images,
                                IMG_WIDTH, IMG_HEIGHT, num_channels), where num_images is the total number of loaded
                                images, IMG_WIDTH and IMG_HEIGHT are the dimensions of each resized image, and
                                num_channels is the number of color channels (e.g., 3 for RGB).
        labels (numpy.ndarray): A NumPy array containing the corresponding labels for the loaded images. The shape
                                of the array is (num_images,), where num_images is the total number of loaded images.
    """
    images, labels = [], []

    for dirpath, _, filenames in os.walk(data_dir):
        if filenames:
            label = int(os.path.basename(dirpath))

        for filename in filenames:
            image = cv2.imread(os.path.join(dirpath, filename))
            resized_image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))

            images.append(resized_image)
            labels.append(label)

    images = np.array(images)
    labels = np.array(labels)        

    return images, labels
    
def get_model():
    """
    Creates and compiles a convolutional neural network (CNN) model for image classification.

    Returns:
        model (tensorflow.keras.models.Sequential): A compiled instance of a CNN model.
    """
    model = tf.keras.models.Sequential([
        # Convolutional layer 1
        tf.keras.layers.Conv2D(200, (7, 7), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        # Max-pooling layer 1
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),

        # Convolutional layer 2
        tf.keras.layers.Conv2D(250, (4, 4), activation='relu'),
        # Max-pooling layer 2
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),

        # Flatten layers
        tf.keras.layers.Flatten(),
        
        # Add a dense hidden layer with 400 units and 50% dropout
        tf.keras.layers.Dense(400, activation='relu'),
        tf.keras.layers.Dropout(0.5),

        # Add a dense output layer for all the labels `NUM_CATEGORIES`
        tf.keras.layers.Dense(NUM_CATEGORIES, activation='softmax')
    ])

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

if __name__ == "__main__":
    main()
