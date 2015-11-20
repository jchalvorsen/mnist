from solve2048 import *
#import pickle
import pickle
import os


def restore_pickled_states(filename):
    load_file = open(filename, 'rb')
    states = pickle.load(load_file)
    moves = pickle.load(load_file)
    return states, moves

def pickle_training_data(states, moves, filename):
    # Ensure we get a unique name to save to
    
    #counter = 1
    #save_file_name = filename + '-' + str(counter) 
    #while os.path.isfile(save_file_name):
    #    counter += 1
    #    save_file_name = filename + '-' + str(counter) 
        

    # Actually dump it to the file
    save_file = open(filename, 'wb')  # 'x' means we need a new file
    
    pickle.dump(states, save_file, -1)
    pickle.dump(moves, save_file, -1)


def run_once_save(depth):
    #window = GameWindow()
    board = np.zeros((GRID_LEN, GRID_LEN), dtype=np.int)
    place_new_tile(board)
    #window.update_view(board)
    
    states = []
    moves = []
    while True:
        new_board, m = expectimax(board, depth)
        states.append(board)
        moves.append(m)
        
        
        board = new_board
        #window.update_view(board)
        place_new_tile(board)
        #window.update_view(board)
        if is_game_over(board):
            break

    #window.destroy()
    return states, moves


def train_one_board():    
    states, moves = run_once_save(3)
    
    if states[-1].max() < 1000:
        return 0
    

    old_states = []
    old_moves = []

    filename = 'test'
    if os.path.isfile(filename):
        old_states, old_moves = restore_pickled_states(filename)
        os.remove(filename)

    pickle_training_data(states + old_states, moves + old_moves, filename)
    return len(states + old_states)
    
    
    
n = 0
while n < 100000:
    n = train_one_board()
    print(n)
        
    
    
    

