import numpy as np
import scipy
import matplotlib
import math

import matplotlib.pyplot as plt
from random import randint
import random

def main():
	fig = plt.figure()

	ax = fig.add_subplot(111)
	ax.set_title('Exercise 6-1a')
	ax.plot(0.1, 0.1, 'bo')
	ax.plot(4, 4, 'bo')
	ax.spines['left'].set_position('zero')
	ax.spines['right'].set_color('none')
	ax.spines['bottom'].set_position('zero')
	ax.spines['top'].set_color('none')

	# remove the ticks from the top and right edges
	ax.xaxis.set_ticks_position('bottom')
	ax.yaxis.set_ticks_position('left')

	plt.show()

if __name__ == "__main__":
	main()
