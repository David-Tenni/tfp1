import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
# remove annoying logs
os.environ['TF_CCP_MIN_LOG_LEVEL'] = '2'

# stopping tensor flow from allocating all memory on the gpu
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)



# images of digits between 0 and 1, they are greyscale with only 1 chanel pixels are 28 by 28

# loading dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# flatten to have one long column
x_train = x_train.reshape(-1, 28*28).astype("float32") / 255.0
x_test = x_test.reshape(-1,28*28).astype('float32') / 255.0

