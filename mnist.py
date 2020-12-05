import numpy
import neural
from tqdm import tqdm
import matplotlib.pyplot

# params
input_nodes = 28 * 28
hidden_layers = 3
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.025
epochs = 15

# initialize network
nn = neural.NeuralNetwork(input_nodes, hidden_layers, hidden_nodes, output_nodes, learning_rate)

# read training data
training_data_file = open("mnist_dataset/mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# train for all training data
loss = []
for e in range(epochs):
    print("epoch = ", e+1)
    for record in tqdm(training_data_list):
        all_values = record.split(',')
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(record[0])] = 0.99
        loss.append(nn.train(inputs, targets))
    print("training loss = ", numpy.average(loss))

    # test data
    test_data_file = open("mnist_dataset/mnist_test.csv", 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    # test for all test data
    scorecard = []
    for record in test_data_list:
        all_values = record.split(',')
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        label = int(all_values[0])
        prediction = numpy.argmax(nn.query(inputs))
        if(label == prediction):
            scorecard.append(1)
        else:
            scorecard.append(0)

    scorecard_array = numpy.asarray(scorecard)
    print("performance = ", scorecard_array.sum() / scorecard_array.size)