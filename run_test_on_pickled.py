
from os import listdir
from os.path import isfile, join
from build_ann import *
import scipy
from math import log, ceil


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
    onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
    onlyfiles.sort()


    #filename = 'pickled/30.200-1'
    nets = ['25.20', '25.40.20', '25.60']

    for filename in onlyfiles:
        for net in nets:
            if filename.contains(net):
                print(filename)
        #print(filename)
        #ANN, time = restore_neural_net(mypath + '/' + filename)
        #percentage = test_network(ANN)  # optional parameter to test on other set
        #print("Percentage:", percentage, "Pickled_net:", filename, ", using", time, "seconds during training.")
        
        
test_all()
random = [5, 1, 2, 4]
results = [6, 1, 6, 3]

t, p = scipy.stats.ttest_ind(random, results)
print(p)
print(max(0,min(7, ceil(-log(p,10)))))
