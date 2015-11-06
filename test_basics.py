__author__ = 'Jon Christian'

from mnist_basics import *
import time


show_avg_digit(3)



data, number = load_mnist()

print(data[0])
show_digit_image(data[0])
print(number[0])

flat = flatten_image(data[0])

print(flat)

show_digit_image(reconstruct_image(flat))

time.sleep(2)

quicktest()