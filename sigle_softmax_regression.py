from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)
#print(mnist.train.images.shape, mnist.train.labels.shape)
#print(mnist.test.images.shape, mnist.test.labels.shape)
#print(mnist.validation.images.shape, mnist.validation.labels.shape)

sess = tf.InteractiveSession() # default session
x = tf.placeholder(tf.float32,[None,784]) # input

W = tf.Variable(tf.zeros([784,10])) # weight
b = tf.Variable(tf.zeros([10]))     # biases
y_hat = tf.nn.softmax(tf.matmul(x,W)+b)     #hypothesis function
y = tf.placeholder(tf.float32,[None,10])    # real data

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_hat), reduction_indices=[1])) # loss function

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy) # Use Gd to minimize the loss

tf.global_variables_initializer().run() 

for i in range(1000):     # Mini batch:512
    batch_xs, batch_ys = mnist.train.next_batch(512)
    train_step.run({x: batch_xs, y: batch_ys})

correct_prediction  =tf.equal(tf.argmax(y,1), tf.argmax(y_hat,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # Accurracy
print(accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

sess.close()
