import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
# remove annoying logs
os.environ['TF_CCP_MIN_LOG_LEVEL'] = '2'

# stopping tensor flow from allocating all memory on the gpu
#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)



# images of digits between 0 and 1, they are greyscale with only 1 chanel pixels are 28 by 28

# loading dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# flatten to have one long column
x_train = x_train.reshape(-1, 28*28).astype("float32") / 255.0
x_test = x_test.reshape(-1,28*28).astype('float32') / 255.0

# Sequential API (convinen, not flexible) if you only need 1 input to 1 output its great otherwise look elsewhere.

model = keras.Sequential(
    [
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(10),
    ]
)

# setting up network configurations, loss functions, optimisers, metrics
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(lr=0.001),
    metrics=["accuracy"],
)

#specifying the training of the netwrok
model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=2)

model.evalute(x_test, y_test, batch_size=32, verbose=2)