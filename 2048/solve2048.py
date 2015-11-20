#! /usr/bin/python
#from visuals import GameWindow
import time
import numpy as np
import random
import math
#from numba import jit, int32
from collections import Counter

GRID_LEN = 4

# grad = np.array(np.mat('7 6 5 4; 6 5 4 3; 5 4 3 2; 4 3 2 1'))
# grad = np.array(np.mat('10 9 8 7; 9 6 5 4; 8 5 3 2; 7 4 2 1'))

""" "optimal" weights gotten from:
https://codemyroad.wordpress.com/2014/05/14/2048-ai-the-intelligent-bot/ """
grad = np.array(np.mat('0.135759  0.121925   0.102812  0.099937;'
                       '0.0997992 0.0888405  0.076711  0.0724143;'
                       '0.060654  0.0562579  0.037116  0.0161889;'
                       '0.0125498 0.00992495 0.00575871 0.003355193'))


GRADIENTS = [grad, np.rot90(grad, 1), np.rot90(grad, 2), np.rot90(grad, 3)]

""" Move left and move methods gotten (at least heavily inspired) from
http://flothesof.github.io/2048-game.html """
def place_new_tile(board):
    i, j = (board == 0).nonzero()
    if i.size != 0:
        rnd = random.randint(0, i.size - 1)
        board[i[rnd], j[rnd]] = 2 * (random.random() > .9) + 2


#@jit(int32[4](int32[4]), nopython=True)
def move_left(col):
    new_col = np.zeros((GRID_LEN), dtype=col.dtype)
    for k in range(GRID_LEN):
        j = 0
        previous = None
        for i in range(GRID_LEN):
            if col[i] != 0:  # number different from zero
                if previous is None:
                    previous = col[i]
                else:
                    if previous == col[i]:
                        # Last number we met is the same, increment number and delete previous so we dont double-merge
                        new_col[j] = 2 * col[i]
                        j += 1
                        previous = None
                    else:
                        # Two different numbers, update previous
                        new_col[j] = previous
                        j += 1
                        previous = col[i]
        if previous is not None:
            # Push element to the left
            new_col[j] = previous
    return new_col


def move(board, direction):
    # 0: left, 1: up, 2: right, 3: down
    new_board = board.copy()
    rotated_board = np.rot90(board, direction)
    for i in range(len(board)):
        new_board[i][:] = move_left(rotated_board[i][:])
    return np.rot90(np.array(new_board), 4-direction)


def evaluate(board):
    sums = [np.multiply(board, g).sum() for g in GRADIENTS]
    return max(sums)


def is_game_over(board):
    # If we have a non-filled board, we have at least one correct move:
    if (board == 0).any():
        return False
    # Do all moves and check if they are the same as parent
    moves = [0, 1, 2, 3]
    for m in moves:
        clone = move(board, m)
        if not (clone == board).all():
            return False
    return True


def get_all_tile_placements(board):
    succ = []
    # Get zero x- and y-indices
    i, j = (board == 0).nonzero()
    for k in range(i.size):
        clone = board.copy()
        clone[i[k]][j[k]] = 2
        succ.append(clone)
    return succ


def expectimax(board, depth):
    moves = [0, 1, 2, 3]
    best_child = None
    best_score = float('-inf')
    best_move = 0
    for m in moves:
        clone = move(board, m)
        if not (clone == board).all():
            score = chance_play(clone, depth - 1)
            if score > best_score:
                best_child = clone
                best_move = m
                best_score = score
    return best_child, best_move


def chance_play(board, depth):
    # "Opponents" move
    # Terminal test: check if there is room to place a new tile
    if (board != 0).all() or depth is 0:
        return evaluate(board)
    score = 0
    succ = get_all_tile_placements(board)
    for b in succ:
        score += max_play(b, depth - 1) / len(succ)
    return score


def max_play(board, depth):
    # Our move
    if is_game_over(board) or depth is 0:
        return evaluate(board)
    moves = [0, 1, 2, 3]
    best_score = float('-inf')
    for m in moves:
        clone = move(board, m)
        score = chance_play(clone, depth - 1)
        if score > best_score:
            best_score = score
    return best_score


def run_once(depth):
    window = GameWindow()
    board = np.zeros((GRID_LEN, GRID_LEN), dtype=np.int)
    place_new_tile(board)
    window.update_view(board)

    while True:
        board, move = expectimax(board, depth)
        window.update_view(board)
        place_new_tile(board)
        window.update_view(board)
        if is_game_over(board):
            score = np.amax(board)
            time.sleep(10)
            window.destroy()
            return score

    time.sleep(5)
    window.destroy()


def main():
    # random.seed(0)
    count = Counter()
    depth = 5
    for k in range(5):
        start = time.time()
        score = run_once(depth)
        end = time.time()
        s2 = 3 * math.log(score / 16, 2)
        count[score] += 1
        print("Got maximum tile of", score, "with a value of", s2, "using", end - start, "seconds.")

    print(count)

#main()


""" Below: Code for using minimax
def alpha_beta_search(board, depth):
    moves = [0, 1, 2, 3]
    best_child = None
    best_score = 0
    for m in moves:
        clone = move(board, m)
        if not (clone == board).all():
            score = min_value(clone, float('-inf'), float('inf'), depth - 1)
            if score > best_score:
                best_child = clone
                best_score = score
    return best_child


def max_value(board, alpha, beta, depth):
    # Terminal test:
    if is_game_over(board) or depth is 0:
        return evaluate(board)
    moves = [0, 1, 2, 3]
    v = float('-inf')
    for m in moves:
        child = move(board, m)
        v = max(v, min_value(child, alpha, beta, depth - 1))
        if v >= beta:
            return v
        alpha = max(v, alpha)
    return v


def min_value(board, alpha, beta, depth):
    # Terminal test:
    if is_game_over(board) or depth is 0:
        return evaluate(board)
    v = float('inf')
    for child in get_all_tile_placements(board):
        v = min(v, max_value(child, alpha, beta, depth - 1))
        if v <= alpha:
            return v
        beta = min(v, beta)
    return v
"""
