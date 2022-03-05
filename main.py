import os
import tensorflow as tf

# remove annoying logs
os.environ['TF_CCP_MIN_LOG_LEVEL'] = '2'

# stopping tensor flow from allocating all memory on the gpu
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

