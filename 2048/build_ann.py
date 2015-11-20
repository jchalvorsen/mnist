__author__ = 'Jon Christian'

import theano
import numpy as np
import theano.tensor as T
import theano.tensor.nnet as Tann
#import pickle
import sys
import os.path
#from mnist_basics import *

# The reduce function was removed in Python 3.0, so just use this handmade version.
def kd_reduce(func,seq):
    res = seq[0]
    for item in seq[1:]:
        res = func(res,item)
    return res

def flatten_image(image_array):
    def flatten(a,b): return a + b
    return kd_reduce(flatten, image_array.tolist())


class ANN():

    # nb = # bits, nh = # hidden nodes (in the single hidden layer)
    # lr = learning rate

    def __init__(self, structure, lr=.2):
        self.lrate = lr
        self.structure = structure
        self.build_ann(structure)
        
    def add_cases(self, cases):
        self.cases = cases
        
    def add_classifications(self, classifications):
        self.classifications = classifications
        compare = []
        for number in self.classifications:
            values = np.zeros(4)
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
        self.params = w+b
        gradients = T.grad(error,self.params)
        backprop_acts = [(p, p - self.lrate*g) for p,g in zip(self.params,gradients)]
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
        results = [];np.zeros((n, 1), dtype=np.int8)
        for i in range(n):
            test = self.predictor(testset[i])
            #print(test, test.argmax())
            results.append(test.argmax())
        return results


def test_network(nn, data, numbers):
    # Get elements from test set:
    #data, numbers = load_mnist(testset)
    flats = [flatten_image(data[i]/255) for i in range(len(data))]
    results = nn.blind_testing(flats)
    
    mysum = 0
    for res, num in zip(results, numbers):
        if res == num: mysum +=1 
    
    
    #print(results,numbers)
    
    return mysum/len(data)*100
    







    


