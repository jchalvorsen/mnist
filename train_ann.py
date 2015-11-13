from build_ann import *


def train(argv):
    epochs = int(argv[1]) if argv[1] else 2
    inner_structure = list(map(int,argv[2:]))
    filename = 'pickled/' + str(epochs) + '.' + '.'.join(map(str, inner_structure))
    
    ANN = train_network(inner_structure, epochs)
    pickle_neural_net(ANN, filename)
    
    percentage = test_network(ANN)
    print("Percentage:", percentage, ", Epochs:", epochs, ", Inner_structure", inner_structure)
    
    
train(sys.argv)
