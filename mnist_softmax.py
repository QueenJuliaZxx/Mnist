
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import data
#from tensorflow.examples.tutorials.mnist import input_data
import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

sess = tf.InteractiveSession()

# Create the model
x = tf.placeholder(tf.float32, [None, 784])
y_true = tf.placeholder(tf.float32, [None, 10])


W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y_predict = tf.nn.softmax(tf.matmul(x, W) + b)



# Define loss and optimizer
cross_entropy = -tf.reduce_sum(y_true * tf.log(y_predict))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# Train:compare with mnist_test.py
tf.initialize_all_variables().run()
for i in range(1000):
  #you can compare with mnist_test.py
  batch_xs, batch_ys = mnist.train.next_batch(100)
  train_step.run({x: batch_xs, y_true: batch_ys})

# Test trained model
correct_prediction = tf.equal(tf.argmax(y_true, 1), tf.argmax(y_predict, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval({x: mnist.test.images, y_true: mnist.test.labels}))
