__author__ = 'Jon Christian'

import theano
import numpy as np
import theano.tensor as T
import theano.tensor.nnet as Tann
import matplotlib.pyplot as plt
import pickle
import sys
from mnist_basics import *

class ANN():

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
            #print('In epoch number:', i)
            error = 0
            for i in range(len(self.cases)):
                error += self.trainer(self.cases[i], self.compare[i])
            errors.append(error)
        return errors

    def blind_testing(self, testset):
        n = len(testset)
        results = numpy.zeros((n, 1), dtype=numpy.int8)
        for i in range(n):
            results[i] = self.predictor(testset[i]).argmax()
        return results


def train_network(inner_structure, epochs):
    data, numbers = load_mnist()

    flats = [flatten_image(data[i]/255) for i in range(len(data))]

    dim_in = len(flats[0])
    dim_out = 10
    
    nn = ANN([dim_in] + inner_structure + [dim_out])
    nn.add_cases(flats)
    nn.add_classifications(numbers)

    errors = nn.do_training(epochs)

    #print(errors)
    return nn


def test_network(nn):
    # Get elements from test set:
    data, numbers = load_mnist('testing')
    flats = [flatten_image(data[i]/255) for i in range(len(data))]
    results = nn.blind_testing(flats)
    mysum = np.sum(results == numbers)
    return mysum/len(data)*100
    



def main(argv):
    epochs = int(argv[1]) if argv[1] else 2
    inner_structure = list(map(int,argv[2:]))
    
    ANN = train_network(inner_structure, epochs)
    percentage = test_network(ANN)
    #print("Got ", percentage, "percent correct")
    print("Percentage:", percentage, ", Epochs:", epochs, ", Inner_structure", inner_structure)

main(sys.argv)
    


