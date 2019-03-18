import tensorflow as tf

# Input data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)

sess = tf.InteractiveSession() # default session

# Weight
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
# Bias
def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
# Convolution
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
# Pooling
def max_pool_f2s2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

# Input and True label
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None, 10])
x2d = tf.reshape(x, [-1,28,28,1])

# Forward propagation
# First layer
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
a_conv1 = tf.nn.relu(conv2d(x2d,W_conv1) + b_conv1)
a_pool1 = max_pool_f2s2(a_conv1)
# Second layer
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
a_conv2 = tf.nn.relu(conv2d(a_pool1,W_conv2) + b_conv2)
a_pool2 = max_pool_f2s2(a_conv2)
# FC layer
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
a_pool2_flat = tf.reshape(a_pool2,[-1,7*7*64]) # 2d to 1d
a_fc1 = tf.nn.relu(tf.matmul(a_pool2_flat,W_fc1) + b_fc1)
# Dropoout
keep_prob = tf.placeholder(tf.float32)
a_fc1_drop = tf.nn.dropout(a_fc1, keep_prob)
# Softmax
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
y_hat = tf.nn.softmax(tf.matmul(a_fc1_drop,W_fc2) + b_fc2)

# Backward propagation
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y*tf.log(y_hat),reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# Evaluation
correct_prediction = tf.equal(tf.argmax(y_hat,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.global_variables_initializer().run()
for i in range(20000):
    batch = mnist.train.next_batch(64)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y:batch[1], keep_prob:1.0})
        print('step %d: training accuracy is %g'%(i, train_accuracy))
    train_step.run(feed_dict={x:batch[0], y:batch[1],keep_prob:0.5})
test_accuracy = accuracy.eval(feed_dict={x:mnist.test.images, y:mnist.test.labels, keep_prob:1.0})
print('test accuracy is %g'%test_accuracy)
sess.close()
