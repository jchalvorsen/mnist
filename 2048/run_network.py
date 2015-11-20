from build_ann import *
import pickle
import time
import scipy

from solve2048 import *



def restore_pickled_states(filename):
    load_file = open(filename, 'rb')
    states = pickle.load(load_file)
    moves = pickle.load(load_file)
    return states, moves




def train_network(inner_structure, epochs):
    data, numbers = restore_pickled_states('test')

    flats = [flatten_image(data[i]/255) for i in range(len(data))]

    dim_in = len(flats[0])
    dim_out = 4
    
    nn = ANN([dim_in] + inner_structure + [dim_out])
    nn.add_cases(flats)
    nn.add_classifications(numbers)

    errors = nn.do_training(epochs)

    #print(errors)
    return nn, errors
    
    
def t_train(epochs, inner_structure):
    
    #filename = 'pickled/' + str(epochs) + '.' + '.'.join(map(str, inner_structure))
    
    t1 = time.time()
    nn, errors = train_network(inner_structure, epochs)
    t2 = time.time()
    #print(errors)
    #pickle_neural_net(ANN, filename, t2-t1)
    data, numbers = restore_pickled_states('test')
    #data = data[0:1000]
    #numbers = numbers[0:1000]
    percentage = test_network(nn, data, numbers)
    print("Percentage:", percentage, ", Epochs:", epochs, ", Inner_structure", inner_structure)
    return nn
    

    


def run_nn(ann):
    board = np.zeros((GRID_LEN, GRID_LEN), dtype=np.int)
    place_new_tile(board)
    
    moves = range(4)
    while True:
        print(board)
        flat = flatten_image(board)
        test = ann.predictor(flat)
        m = test.argmax()
        new_board = move(board, m)
        #print new_board == board, (new_board==board).all()
        
        #while (new_board == board).all():
        #    test = ann.predictor(flat)
        #    m = test.argmax()
        #    new_board = move(board, m)
        
        board = new_board

        place_new_tile(board)
        if is_game_over(board):
            score = np.amax(board)
            #time.sleep(10)
            break

    return np.amax(board)
    #return states, moves
epochs = 4
inner_structure = [12]     
nn =  t_train(epochs, inner_structure)
#score = run_nn(nn)
#print(score)    

 

