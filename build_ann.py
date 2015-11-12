__author__ = 'Jon Christian'

import theano
import numpy as np
import theano.tensor as T
import theano.tensor.nnet as Tann
import matplotlib.pyplot as plt
from mnist_basics import *

class autoencoder():

    # nb = # bits, nh = # hidden nodes (in the single hidden layer)
    # lr = learning rate

    def __init__(self, structure, lr=.1):
        self.lrate = lr
        self.build_ann(structure)
        
    def add_cases(self, cases):
        self.cases = cases
        
    def add_classifications(self, classifications):
        self.classifications = classifications
        compare = []
        for number in self.classifications:
            values = np.zeros(10)
            values[number] = 1
            compare.append(values)
        self.compare = compare

    def build_ann(self,structure): 
        w = []      # list of weight matrices
        b = []      # list of bias vectors
        x = []      # list of node values, first is input, last is output
        input = T.dvector('input')
        compare = T.dvector('compare')
        x.append(input)
        for i in range(len(structure)-1):
            dim1 = structure[i]
            dim2 = structure[i+1]
            w.append(theano.shared(np.random.uniform(-.1,.1,size=(dim1,dim2))))
            b.append(theano.shared(np.random.uniform(-.1,.1,size=dim2)))
            x.append(Tann.sigmoid(T.dot(x[i],w[i]) + b[i]))

        error = T.sum((x[-1]-compare)**2)
        params = w+b
        gradients = T.grad(error,params)
        backprop_acts = [(p, p - self.lrate*g) for p,g in zip(params,gradients)]
        self.predictor = theano.function([input],x[-1])
        self.trainer = theano.function([input, compare],error,updates=backprop_acts)

    def do_training(self, epochs=100):
        errors = []
        for i in range(epochs):
            print('In epoch number:', i)
            error = 0
            for i in range(len(self.cases)):
                error += self.trainer(self.cases[i], self.compare[i])
            errors.append(error)
        return errors

    def do_testing(self, testset):
        n = len(testset)
        results = numpy.zeros((n, 1), dtype=numpy.int8)
        for i in range(n):
            results[i] = self.predictor(testset[i]).argmax()
        return results


data, numbers = load_mnist()

flats = [flatten_image(data[i]/255) for i in range(len(data))]

dim = len(flats[0])

auto = autoencoder([dim, 20, 10])


auto.add_cases(flats)
auto.add_classifications(numbers)

errors = auto.do_training(2)


print(errors)

# Get elements from test set:
number_to_test = 10000
data, numbers = load_mnist('testing')
data = data[0:number_to_test]
numbers = numbers[0:number_to_test]
flats = [flatten_image(data[i]/255) for i in range(len(data))]


results = auto.do_testing(flats)
#print(results)
#print(np.asarray(numbers))

mysum = np.sum(results == numbers)
#print(mysum)

"""
for i in range(number_to_test):
    flat = flats[i]
    value = numbers[i][0]
    result = auto.predictor(flat)
    index = result.argmax()
    if index == value:
        correct += 1
    #print('Guessed', index, ", correct was", value)
"""
print("Got ", mysum/number_to_test*100, "percent correct")






