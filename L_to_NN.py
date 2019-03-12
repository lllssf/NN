import numpy
# sigmoid function expit()
from scipy.special import expit
# library for plotting arrays
import matplotlib.pyplot as plt
# show the picture
import pylab


# neural network class defination
class neuralNetwork(object):
    # initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # set learning rate
        self.lr = learningrate

        # link weight matrices, winput_hidden and whidden_output
        # weights inside the array are w_i_j, where link is form node i to the node j in the next layer
        # w11 w21 ...
        # w12 w22 etc
        # O = W . I, 正态概率分布采样权重
        self.wih = (numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes)))
        self.who = (numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes)))

        # 为了方便随时更改激活函数，在初始化部分创建激活函数，用lambda来创建函数，方便又快捷，也被称为匿名函数
        # acitivation function is the sigmoid function
        self.activation_function = lambda x: expit(x)
        pass

    # train the neural network
    def train(self, inputs_list, target_list):
        # convert inputs_list and target_list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(target_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        # error is the (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = numpy.dot((self.who).T, output_errors)

        # update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot(output_errors * final_outputs * (1.0 - final_outputs),
                                        numpy.transpose(hidden_outputs))
        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * numpy.dot(hidden_errors * hidden_outputs * (1.0 - hidden_outputs),
                                        numpy.transpose(inputs))
        pass

    # query the neural network.给输入得到输出
    def query(self, inputs_list):
        # convert inputs list to 2d array，转置为列向量
        inputs = numpy.array(inputs_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


# END

# create object
# number of input, hidden, output nodes
input_nodes = 784
hidden_nodes = 200
output_nodes = 10
# learning rate is 0.2(sweetpoint)
learning_rate = 0.2
# creat instance of neural network
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# load the MNIST training data CSV file into a list
training_data_file = open("mnist_database/mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# train the neural network
# epochs is the number of times the training data set is used
epochs = 2
for e in range(epochs):
    # go through all records in the training data set
    for record in training_data_list:
        # split the record by the ',' commas
        all_values = record.split(',')
        # scale and shift the inputs
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # creat the target output values(all 0.01, except the disired label which is 0.99)
        targets = numpy.zeros(output_nodes) + 0.01
        # all_values[0] is the target label for this record
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        pass
    pass

# test the neural network
# load the minist test data CSV into a list
test_data_file = open("mnist_database/mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

#scorecard for how well the network performs, initially empty
scorecard = []

# go through a;; the records in the test data base
for record in test_data_list:
    #split the record by the ',' commas
    all_values = record.split(',')
    # correct answer is the first value
    correct_label = int(all_values[0])
    # scale and shif the inputs
    inputs = (numpy.asfarray(all_values[1:])/255*0.99)+0.01
    #query the network
    outputs = n.query(inputs)
    # the index of the highest value correspond to the label
    label = numpy.argmax(outputs)
    # append correct or incorrect to scorecard
    if(label == correct_label):
        #network's answer matches correct answer, add 1 to scorecard
        scorecard.append(1)
    else:
        #network's answer doesn't match correct answer, add 0 to scorecard
        scorecard.append(0)
        pass
    pass
#print(scorecard)

#calculate the performance score, the fraction of correct answers
scorecard_array = numpy.asfarray(scorecard)
print("performance = ",scorecard_array.sum()/scorecard_array.size)
