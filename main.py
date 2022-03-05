import os

# remove annoying logs
os.environ['TF_CCP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

print(tf.__version__)

# initialisation of tensors

# manual initialisation
# x = tf.constant(4, shape=(1, 1), dtype=tf.float32)
# x = tf.constant([[1, 2, 3], [4, 5, 6]])

# auto initialisation below creates 3 by 3 matrix
t = tf.range(start=1, limit=10, delta=2)
u = tf.range(9)
v = tf.random.uniform((1, 3), minval=0, maxval=55)
w = tf.random.normal((3, 3), mean=0, stddev=1)
x = tf.ones((3, 3))
y = tf.zeros((2, 3))
z = tf.eye(3)  # I for identity matrix (eye)

print(z, y, x, w, v, u, t)

# mathematics Operations

# indexing of a tensor

# reshaping a tensor
