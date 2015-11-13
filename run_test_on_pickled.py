from build_ann import *



filename = 'pickled/2.20-3'
ANN = restore_neural_net(filename)
percentage = test_network(ANN)
print("Percentage:", percentage, "Pickled_net:", filename)
