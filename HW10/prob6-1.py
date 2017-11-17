import numpy
import scipy
import matplotlib

import matplotlib.pyplot as plt
from random import randint
import random

if __name__ == "__main__":
	dataPOS = []
	dataNEG = []
	datax1NEG = [1,0,0,-1]
	datax2NEG = [0,1,-1,0]
	datax1POS = [2,-2,0]
	datax2POS = [0,0,-2]
	ys = [-1,-1,-1,-1,1,1,1]

	negOnes = []
	i = 0
	while (i < len(datax1NEG)):
		negOnes.append(-1)
		i += 1
	posOnes = []
	i = 0
	while (i < len(datax1POS)):
		posOnes.append(1)
		i += 1

	dataNEG.append(negOnes)
	dataNEG.append(datax1NEG)
	dataNEG.append(datax2NEG)
	dataPOS.append(posOnes)
	dataPOS.append(datax1POS)
	dataPOS.append(datax2POS)

	fig = plt.figure()
	plt.plot(dataPOS[1], dataPOS[2], 'bo', label = "1")
	plt.plot(dataNEG[1], dataNEG[2], 'rx', label = "-1")
	fig.suptitle('Nearest Neighbor', fontsize = 20)
	plt.xlabel('x1', fontsize = 18)
	plt.ylabel('x2', fontsize = 18)
	plt.legend(loc = 'upper right')
	plt.axis([-2.5,2.5,-2.5,2.5])
	plt.show()