from build_ann import *
import time


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
    
    
def pickle_neural_net(nn, filename, time):
    # Ensure we get a unique name to save to
    
    counter = 1
    save_file_name = filename + '-' + str(counter) 
    while os.path.isfile(save_file_name):
        counter += 1
        save_file_name = filename + '-' + str(counter) 
        

    # Actually dump it to the file
    save_file = open(save_file_name, 'wb')  # 'x' means we need a new file
    
    pickle.dump(time, save_file, -1)
    pickle.dump(nn.lrate, save_file, -1)
    pickle.dump(nn.structure, save_file, -1)
    for arg in nn.params:
        pickle.dump(arg.get_value(borrow=True), save_file, -1)


def train_with_arguments(argv): #train(sys.argv)

    epochs = int(argv[1]) if argv[1] else 2
    inner_structure = list(map(int,argv[2:]))
    filename = 'pickled/' + str(epochs) + '.' + '.'.join(map(str, inner_structure))
    
    t1 = time.time()
    ANN = train_network(inner_structure, epochs)
    t2 = time.time()
    pickle_neural_net(ANN, filename, t2-t1)
    
    percentage = test_network(ANN)
    print("Percentage:", percentage, ", Epochs:", epochs, ", Inner_structure", inner_structure)
    
    
def t_train(epochs, inner_structure):
    
    filename = 'pickled/' + str(epochs) + '.' + '.'.join(map(str, inner_structure))
    
    t1 = time.time()
    ANN = train_network(inner_structure, epochs)
    t2 = time.time()
    pickle_neural_net(ANN, filename, t2-t1)
    
    percentage = test_network(ANN)
    print("Percentage:", percentage, ", Epochs:", epochs, ", Inner_structure", inner_structure)
    
    
epochs = 2
inner_structure = [20]    
number_of_nets = 2    
for i in range(number_of_nets): t_train(epochs, inner_structure)





