import numpy as np
import scipy.special
import matplotlib.pyplot


class NeuralNetwork:
    def __init__(self, inodes, hnodes, onodes, learningrade, epochs):
        self.inodes = inodes
        self.hnodes = hnodes
        self.onodes = onodes
        self.epochs = epochs
        self.learningrade = learningrade
        self.wih = np.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))
        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self, input_list, target_list):
        """Preparing input data and target values"""
        inputs = np.array(input_list, ndmin=2).T
        targets = np.array(target_list, ndmin=2).T

        """Getting data on hidden and final layers"""
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        """Calculating errors"""
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)

        """Updating waits"""
        self.who += self.learningrade * np.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                               np.transpose(hidden_outputs))
        self.wih += self.learningrade * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                               np.transpose(inputs))

    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


class Data:
    def __init__(self, path):
        self.path = path
        with open(self.path, 'r') as all_data:
            self.data = all_data.readlines()
        self.input = []
        self.target = []
        self.mark = 0

    def transform(self, out_nodes, row):
        self.mark = int(self.data[row].split(',')[0])
        self.input = (np.asfarray(self.data[row].split(',')[1:]) / 255.0 * 0.99) + 0.01
        self.target = np.zeros(out_nodes) + 0.01
        self.target[int(self.data[row][0])] = 0.99


nn = NeuralNetwork(784, 200, 10, 0.1, 5)


train_data = Data('mnist_train.csv')
test_data = Data('mnist_test.csv')
test = []

for epoch in range(nn.epochs):
    for i in range(len(train_data.data)):
        train_data.transform(nn.onodes, i)
        nn.train(train_data.input, train_data.target)


for i in range(len(test_data.data)):
    test_data.transform(nn.onodes, i)
    out = nn.query(test_data.input)
    if np.argmax(out) == test_data.mark:
        test.append(1)
    else:
        test.append(0)
print(test.count(1) / len(test))



