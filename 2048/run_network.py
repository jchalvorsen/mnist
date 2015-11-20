from build_ann import *
import pickle
import time
import scipy

from solve2048 import *
from random_player import *

import requests

def welch(list1, list2):
    params = {"results": str(list1) + " " + str(list2), "raw": "1"}
    resp = requests.post('http://folk.ntnu.no/valerijf/6/', data=params)
    return resp.text

def restore_pickled_states(filename):
    load_file = open(filename, 'rb')
    states = pickle.load(load_file)
    moves = pickle.load(load_file)
    return states, moves



def take_2_log(board, do_log = False):
    if do_log:
        new_board = np.zeros((GRID_LEN, GRID_LEN), dtype=np.double)
        for i in range(GRID_LEN):
            for j in range(GRID_LEN):
                if board[i][j] != 0:    
                    new_board[i][j] = np.log2(board[i][j])/8
        return new_board
    else:
        return board/255
        


def train_network(inner_structure, epochs, do_log):
    data, numbers = restore_pickled_states('test')
    
    data_2 = [take_2_log(d, do_log) for d in data]
    
    flats = [flatten_image(data_2[i]) for i in range(len(data_2))]

    dim_in = len(flats[0])
    dim_out = 4
    
    nn = ANN([dim_in] + inner_structure + [dim_out])
    nn.add_cases(flats)
    nn.add_classifications(numbers)

    errors = nn.do_training(epochs)
    
    return nn, errors
    
    
def t_train(epochs, inner_structure):
    
    t1 = time.time()
    nn, errors = train_network(inner_structure, epochs)
    t2 = time.time()
    data, numbers = restore_pickled_states('test')
    percentage = test_network(nn, data, numbers)
    print("Percentage:", percentage, ", Epochs:", epochs, ", Inner_structure", inner_structure)
    return nn, errors
    

    


def run_nn(ann, do_log):
    board = np.zeros((GRID_LEN, GRID_LEN), dtype=np.int)
    place_new_tile(board)
    
    while True:
        #print(board)
        board_2 = take_2_log(board, do_log)
        flat = flatten_image(board_2)
        #print(flat)
        #print(max(flat))
        predicted = ann.predictor(flat)
        
        m = predicted.argmax()
        new_board = move(board, m)
        
        
        while (new_board == board).all():
            # invalid move, pick another
            predicted[m] = 0
            m = predicted.argmax()
            new_board = move(board, m)

        
        board = new_board

        place_new_tile(board)

        if is_game_over(board):

            break

    return np.amax(board)
    
for i in range(100):    
    do_log = False
    epochs = 4
    inner_structure = [12]     

    nn, errors = train_network(inner_structure, epochs, do_log)
    print(errors)

    scores = []
    randoms = []
    test_number = 50
    for i in range(test_number):
        scores.append(run_nn(nn, do_log))
        randoms.append(run_random())
    print(scores)
    final_score = welch(randoms, scores)
    print(final_score)   

 

