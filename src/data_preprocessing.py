import tensorflow as tf
import numpy as np

def load_and_filter_mnist(digit_1, digit_2):
    """
    Loads the MNIST dataset and filters it to contain only two specified digits.
    This simplifies the 10-class problem into a binary classification problem.
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize pixel values
    x_train, x_test = x_train / 255.0, x_test / 255.0

    def filter_dataset(x, y):
        """Helper to filter images and labels for the two target digits."""
        keep = (y == digit_1) | (y == digit_2)
        x, y = x[keep], y[keep]
        # Remap labels to 0 and 1
        y = np.where(y == digit_1, 0, 1)
        return x, y

    x_train, y_train = filter_dataset(x_train, y_train)
    x_test, y_test = filter_dataset(x_test, y_test)

    # Add a channel dimension for the CNN (e.g., 28x28 -> 28x28x1)
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    
    # Resize images to 14x14 to reduce classical parameters
    x_train = tf.image.resize(x_train, [14,14]).numpy()
    x_test = tf.image.resize(x_test, [14,14]).numpy()

    print(f"--- Data Preprocessing ---")
    print(f"Filtered dataset for digits {digit_1} and {digit_2}.")
    print(f"Training images shape: {x_train.shape}")
    print(f"Testing images shape: {x_test.shape}")
    
    return x_train, y_train, x_test, y_test
