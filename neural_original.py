import numpy
import scipy.special


class NeuralNetwork:
    # initialize neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate
        self.activation_function = lambda x: scipy.special.expit(x)

        # initialize wight -0.5 to 0.5
        self.wih = (numpy.random.rand(self.hnodes, self.inodes) - 0.5)
        self.who = (numpy.random.rand(self.onodes, self.hnodes) - 0.5)
        pass

    # train neural network
    # multiple inputs and outputs
    def train(self, inputs_list, targets_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # calculate outputs of neural network for each multiple inputs at once
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        # errors for each outputs
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # update weight
        self.who = self.who + self.lr * numpy.dot(output_errors * final_outputs * (1 - final_outputs), hidden_outputs.T)
        self.wih = self.wih + self.lr * numpy.dot(hidden_errors * hidden_outputs * (1 - hidden_outputs), inputs.T)

        pass

    # calculate output of neural network
    def query(self, inputs):
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs
