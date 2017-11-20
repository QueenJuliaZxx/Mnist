import tensorflow.python.platform
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
                     'for unit testing.')
flags.DEFINE_integer('max_steps', 1000, 'Number of steps to run trainer.')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')


def main(_):
  # Import data
  mnist = input_data.read_data_sets('MNIST_data/', one_hot=True,
                                    fake_data=FLAGS.fake_data)

  sess = tf.InteractiveSession()

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784], name='x-input')
  W = tf.Variable(tf.zeros([784, 10]), name='weights')
  b = tf.Variable(tf.zeros([10], name='bias'))
  y_true= tf.placeholder(tf.float32, [None, 10], name='y-label')

  # Use a name scope to organize nodes in the graph visualizer
  with tf.name_scope('Wx_b'):
    y_predict = tf.nn.softmax(tf.matmul(x, W) + b)

  # Add summary ops to collect data
  _ = tf.summary.histogram('weights', W)
  _ = tf.summary.histogram('biases', b)
  _ = tf.summary.histogram('y_predict', y_predict)

  # More name scopes will clean up the graph representation
  with tf.name_scope('cross_entropy'):
    cross_entropy = -tf.reduce_sum(y_true * tf.log(y_predict))
    _ = tf.summary.scalar('cross entropy', cross_entropy)
  with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(
        FLAGS.learning_rate).minimize(cross_entropy)
  with tf.name_scope('test'):
    correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_true, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    _ = tf.summary.scalar('accuracy', accuracy)

  # Merge all the summaries and write into /mnist_logs
  merged = tf.summary.merge_all()
  writer = tf.summary.FileWriter('/raid5/ZXX_Project/mnist-master/MNIST_data/mnist_logs', sess.graph_def)
  sess.run(tf.initialize_all_variables())


  # Train

  for i in range(FLAGS.max_steps):
    if i % 10 == 0:  # Record summary data and the accuracy every 10 steps
      if FLAGS.fake_data:
        batch_xs, batch_ys = mnist.train.next_batch(
            100, fake_data=FLAGS.fake_data)
        feed = {x: batch_xs, y_true: batch_ys}
      else:
        feed = {x: mnist.test.images, y_true: mnist.test.labels}
      result = sess.run([merged, accuracy], feed_dict=feed)
      summary_str = result[0]
      acc = result[1]
      writer.add_summary(summary_str, i)
      print('Accuracy at step %s: %s' % (i, acc))
    else:
      batch_xs, batch_ys = mnist.train.next_batch(
          100, fake_data=FLAGS.fake_data)
      feed = {x: batch_xs, y_true: batch_ys}
      sess.run(train_step, feed_dict=feed)

if __name__ == '__main__':
  tf.app.run()
