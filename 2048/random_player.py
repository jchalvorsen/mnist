from solve2048 import *
import random



def run_random():
    board = np.zeros((GRID_LEN, GRID_LEN), dtype=np.int)
    place_new_tile(board)
    
    moves = range(4)
    while True:
        
        m = random.randint(0,3)
        new_board = move(board, m)  
        
        #print new_board == board, (new_board==board).all()
        
        while (new_board == board).all():
            m = random.randint(0,3)
            new_board = move(board, m)
        
        board = new_board

        place_new_tile(board)
        if is_game_over(board):
            score = np.amax(board)
            #time.sleep(10)
            break

    return np.amax(board)
    #return states, moves

score = run_random()
print score
