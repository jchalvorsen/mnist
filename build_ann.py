__author__ = 'Jon Christian'

import theano
import numpy as np
import theano.tensor as T
import theano.tensor.nnet as Tann
import matplotlib.pyplot as plt
from mnist_basics import *


def gen_all_bit_cases(num_bits):
    def bits(n):
        s = bin(n)[2:]
        return [int(b) for b in '0'*(num_bits - len(s))+s]
    return [bits(i) for i in range(2**num_bits)]


class autoencoder():

    # nb = # bits, nh = # hidden nodes (in the single hidden layer)
    # lr = learning rate

    def __init__(self,nb=3,nh=2,nr=3,lr=.1):
        #self.cases = gen_all_bit_cases(nb)
        self.lrate = lr
        self.build_ann(nb,nh,nr,lr)

    def build_ann(self,nb,nh,nr,lr):
        w1 = theano.shared(np.random.uniform(-.1,.1,size=(nb,nh)))
        w2 = theano.shared(np.random.uniform(-.1,.1,size=(nh,nr)))
        input = T.dvector('input')
        values = T.dvector('values')
        b1 = theano.shared(np.random.uniform(-.1,.1,size=nh))
        b2 = theano.shared(np.random.uniform(-.1,.1,size=nr))
        x1 = Tann.sigmoid(T.dot(input,w1) + b1)
        x2 = Tann.sigmoid(T.dot(x1,w2) + b2)
        error = T.sum((x2-values)**2)
        params = [w1,b1,w2,b2]
        gradients = T.grad(error,params)
        backprop_acts = [(p, p - self.lrate*g) for p,g in zip(params,gradients)]
        self.predictor = theano.function([input],[x2,x1])
        self.trainer = theano.function([input, values],error,updates=backprop_acts)

    def do_training(self, epochs=100):
        errors = []
        for i in range(epochs):
            print('In epoch number:', i)
            error = 0
            for i in range(len(self.cases)):
                case = self.cases[i]
                #print(max(case))
                values = np.zeros(10)
                values[self.numbers[i][0]] = 1
                #print(self.numbers[i][0])
                error += self.trainer(case, values)
            #for c in self.cases:
            #    error += self.trainer(c, 7)
            errors.append(error)
        return errors

    def do_testing(self):
        hidden_activations = []
        for c in self.cases:
            _, hact = self.predictor(c)
            hidden_activations.append(hact)
        return hidden_activations


"""
auto = autoencoder(3,2)
print(auto.cases)
errors = auto.do_training(1000)
plt.plot(errors)
plt.show()

case = auto.cases[1]
print(case)
result = auto.predictor(case)
end = result[0]
print(end)
plt.barh(range(len(end)), end)
plt.show()
print(result)
"""




data, numbers = load_mnist()
#data = data[0:10000]
#numbers = numbers[0:10000]


flats = [flatten_image(data[i]/255) for i in range(len(data))]

dim = len(flats[0])

auto = autoencoder(dim, 20, 10)
auto.cases = flats
auto.numbers = numbers

errors = auto.do_training(20)

#plt.plot(errors)
#plt.show()

print(errors)
"""
flat = flats[10]
show_digit_image(reconstruct_image(flat))
print('Classification:', numbers[10])
result = auto.predictor(flat)
print(result)
#plt.barh(range(10),result)
#plt.show()
"""

# Get elements from test set:
number_to_test = 10000
data, numbers = load_mnist('testing')
data = data[0:number_to_test]
numbers = numbers[0:number_to_test]
flats = [flatten_image(data[i]/255) for i in range(len(data))]

correct = 0
for i in range(number_to_test):
    flat = flats[i]
    value = numbers[i][0]
    result, _ = auto.predictor(flat)
    index = result.argmax()
    if index == value:
        correct += 1
    #print('Guessed', index, ", correct was", value)

print("Got ", correct/number_to_test*100, "percent correct")
