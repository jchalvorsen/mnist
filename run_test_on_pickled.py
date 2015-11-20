
from os import listdir
from os.path import isfile, join
from build_ann import *
import scipy
from math import log, ceil
import numpy as np


def restore_neural_net(filename):
    load_file = open(filename, 'rb')
    time = pickle.load(load_file)
    lrate = pickle.load(load_file)
    structure = pickle.load(load_file)
    nn = ANN(structure)
    nn.lrate = lrate
    for arg in nn.params:
        arg.set_value(pickle.load(load_file), borrow=True)
    return nn, time



def test_all():
    mypath = 'pickled'
    nets = ['25.20-','25.60-', '25.200-', '25.40.20-', '25.200.60.20-']
    all_nets = []   
    for net in nets:
        onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) if net in f ]
        all_nets.append(onlyfiles)
        
        
    
    # Calculate correct percentage for each net
    percentages = []
    for net in all_nets:
        p1 =[]
        for filename in net:
            ANN, time = restore_neural_net(mypath + '/' + filename)
            percentage = test_network(ANN, 'training')  # optional parameter to test on other set
            print("Percentage:", percentage, "Pickled_net:", filename, ", using", time, "seconds during training.")
            p1.append(percentage)
        percentages.append(p1)
    
    # Calculate co-p matrix
    n = len(percentages)
    co_p = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            _, co_p[i][j] = scipy.stats.ttest_ind(percentages[i], percentages[j], False)
    print('Indexes:', nets)
    print('Co-p-matrix:')
    print(co_p)
    print('Mean for nets:')
    print([np.mean(l) for l in percentages])
    #for filename in onlyfiles:
        #print(filename)
        #ANN, time = restore_neural_net(mypath + '/' + filename)
        #percentage = test_network(ANN)  # optional parameter to test on other set
        #print("Percentage:", percentage, "Pickled_net:", filename, ", using", time, "seconds during training.")
        
        
test_all()

