import os
import tensorflow as tf

# remove annoying logs
os.environ['TF_CCP_MIN_LOG_LEVEL'] = '2'
# stopping tensor flow from allocating all memory on the gpu
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

print(tf.__version__)

# initialisation of tensors

# manual initialisation
# x = tf.constant(4, shape=(1, 1), dtype=tf.float32)
# x = tf.constant([[1, 2, 3], [4, 5, 6]])

# auto initialisation below creates 3 by 3 matrix
print('Initialisation of Tensors')
a = tf.range(start=1, limit=10, delta=2)
a = tf.cast(a, dtype=tf.float64)
b = tf.range(9)
c = tf.random.uniform((1, 3), minval=0, maxval=55)
d = tf.random.normal((3, 3), mean=0, stddev=1)
e = tf.ones((3, 3))
f = tf.zeros((2, 3))
g = tf.eye(3)  # I for identity matrix (eye)

print("a ", a, "b: ", b, "c :", c, "d: ", d, "e: ", e, "f: ", f, "g: ", g)

# mathematics operations
print('Mathematics Operations')
h = tf.constant([9, 8, 7])
i = tf.constant([1, 2, 3])
# j = tf.add(h, i)
j = h + i  # alternate form of above line it is more convenient
# k = tf.subtract(h, i)
k = h - i  # alternative form of above line, more convenient

# l = tf.divide(h,i)
l = h / i  # simplified form of above

# m = tf.multiply(h, i)
m = h * i  # simplified of above

n = tf.tensordot(h, i, axes=1)  # below is what be considering the manual way of doing it, but it is no longer simpler
# n = tf.reduce_sum(h*i, axis=0)

o = h ** 5

pa = tf.random.normal((2, 3))
pb = tf.random.normal((3, 4))

# p = tf.matmul(pa, pb)
p = pa @ pb  # simplified of above line
print("h: ", h, "i: ", i, "j: ", j, "k :", k, "l :", l, "m: ", m, "n :", n, "o :", o, "p :", p)
# indexing of a tensor
print('indexing of a tensor')
# reshaping a tensor
print("reshaping a tensor")
