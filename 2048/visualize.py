from visuals import *
import numpy as np
import time

outfile = '2048game.npy'
boards = np.load(outfile)
window = GameWindow()

for i in range(len(boards)):
    print(i)
    board = boards[i]
    #print board
    window.update_view(board)
    raw_input()


"""
i = 210
window.update_view(boards[i])
time.sleep(5)
window.update_view(boards[i+1])
"""
time.sleep(5)



window.destroy()
