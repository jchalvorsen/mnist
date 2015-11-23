from build_ann import *
import pickle
import time
import scipy
import copy
import pylab 
import math

from solve2048 import *
from random_player import *
#from visuals import *

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
    board = np.zeros((GRID_LEN, GRID_LEN), dtype=int)
    place_new_tile(board)
    boards = [board]
    
    while True:
        board_2 = take_2_log(board, do_log)
        flat = flatten_image(board_2)
        predicted = ann.predictor(flat)
        
        m = predicted.argmax()
        new_board = move(board, m)
        
        
        while (new_board == board).all():
            # invalid move, pick another
            predicted[m] = 0
            m = predicted.argmax()
            new_board = move(board, m)

        
        board = new_board
        boards.append(copy.deepcopy(board))

        place_new_tile(board)
        boards.append(board)

        if is_game_over(board):

            break

    return np.amax(board), boards


def save_to_file(boards, filename):
    np.save(filename, boards)
    
def test_50():   
    do_log = False
    epochs = 4
    inner_structure = [12]     

    nn, errors = train_network(inner_structure, epochs, do_log)


    # Score both networks + random player k times:
    scores = []
    randoms = []
    k = 50
    for i in range(k):
        scores.append(run_nn(nn, do_log)[0])
        randoms.append(run_random())

    print("Average of score for normal system:", np.average(scores))
    print("Average of score for random system:", np.average(randoms))

    return scores, randoms

def test_all():   
    do_log = False
    epochs = 4
    inner_structure = [12]     

    # Train two different networks with different preprocessing
    nn, errors = train_network(inner_structure, epochs, do_log)
    nn2, errors2 = train_network(inner_structure, epochs, not do_log)
    print(errors)
    print(errors2)

    # Score both networks + random player k times:
    scores = []
    scores2 = []
    randoms = []
    k = 500
    for i in range(k):
        print(i)
        scores.append(run_nn(nn, do_log)[0])
        scores2.append(run_nn(nn2, not do_log)[0])
        randoms.append(run_random())

    print("Average of score for normal system:", np.average(scores))
    print("Average of score for log system:", np.average(scores2))
    print("Average of score for random system:", np.average(randoms))

    print(welch(randoms, scores))


def test_and_save_one():
    do_log = False
    epochs = 4
    inner_structure = [12]     
    nn, errors = train_network(inner_structure, epochs, do_log)
    score, boards = run_nn(nn, do_log)
    #print("Score:", score)
    save_to_file(boards, '2048game')
    return score

 
 
 
for i in range(10): 
    scores, randoms = test_50() 
    p = scipy.stats.ttest_ind(np.log2(scores), np.log2(randoms), equal_var=False)[1]
    p2 = scipy.stats.ttest_ind(scores, randoms, equal_var=False)[1]
    score = max(0, math.ceil(-math.log(p,10)))
    score2 = max(0, math.ceil(-math.log(p2,10)))
    print(p, p2, score, score2)
    scipy.stats.probplot(np.log2(scores), dist="norm", plot=pylab)
    pylab.show() 
#test_and_save_one()

#test_all()    

