from solve2048 import *
import pickle

def restore_pickled_states(filename):
    load_file = open(filename, 'rb')
    states = pickle.load(load_file)
    moves = pickle.load(load_file)
    return states, moves
    
    
def plot_game(states, moves):
    window = GameWindow()
    for board, m in zip(states, moves):
        window.update_view(board)
        #time.sleep(1)
        new_board = move(board, m)
        window.update_view(new_board)
        #time.sleep(1)
        

states, moves = restore_pickled_states('test')
#print(moves)


all_moves =  [0, 0, 0, 0]
n = len(moves)

for move in moves:
    all_moves[move] += 1/n
    
print(all_moves)
print(n)
    
#plot_game(states, moves)
        
