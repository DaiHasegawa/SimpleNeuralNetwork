import numpy
import scipy.special

class NeuralNetwork:
    # initialize neural network
    def __init__(self, inputnodes, hiddenlayers, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate
        self.activation_function = lambda x: scipy.special.expit(x)
        self.layers = hiddenlayers + 2
        self.weights = self.layers - 1
        self.errors = self.layers - 1

        # initialize wight -0.5 to 0.5
        self.w = [0] * self.weights
        self.w[0] = (numpy.random.rand(self.hnodes, self.inodes) - 0.5)
        for i in range(1, self.weights - 1):
            self.w[i] = (numpy.random.rand(self.hnodes, self.hnodes) - 0.5)
        pass
        self.w[self.weights - 1] = (numpy.random.rand(self.onodes, self.hnodes) - 0.5)

    # train neural network
    # multiple inputs and outputs
    def train(self, inputs_list, targets_list):

        targets = numpy.array(targets_list, ndmin=2).T

        # calculate outputs of neural network for each multiple inputs at once
        inputs = [0] * self.layers
        outputs = [0] * self.layers
        inputs[0] = numpy.array(inputs_list, ndmin=2).T
        outputs[0] = inputs[0]
        for i in range(1, self.layers):
            inputs[i] = numpy.dot(self.w[i - 1], outputs[i - 1])
            outputs[i] = self.activation_function(inputs[i])

        # errors for each outputs
        e = [0] * self.errors
        e[self.errors - 1] = targets - outputs[self.layers - 1]
        for i in range(self.errors - 2, -1, -1):
            e[i] = numpy.dot(self.w[i + 1].T, e[i + 1])

        # update weight
        for i in range(0, self.weights):
            self.w[i] = self.w[i] + self.lr * numpy.dot(e[i] * outputs[i + 1] * (1 - outputs[i + 1]), outputs[i].T)

        # calculate training loss
        loss = []
        for i in range(0, self.errors):
            loss.append(numpy.average(e[i]))

        return numpy.average(loss)

    # calculate output of neural network
    def query(self, input_lists):
        inputs = [0] * self.layers
        outputs = [0] * self.layers
        inputs[0] = numpy.array(input_lists, ndmin=2).T
        outputs[0] = inputs[0]
        for i in range(1, self.layers):
            inputs[i] = numpy.dot(self.w[i - 1], outputs[i - 1])
            outputs[i] = self.activation_function(inputs[i])

        return outputs[self.layers - 1]
