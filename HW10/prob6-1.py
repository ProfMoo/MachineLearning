import numpy
import scipy
import matplotlib
import math

import matplotlib.pyplot as plt
from random import randint
import random

def getData():
	dataPOS = []
	dataNEG = []
	datax1POS = [2.,-2.,0]
	datax2POS = [0,0,-2.]
	datax1NEG = [1.,0,0,-1.]
	datax2NEG = [0,1.,-1.,0]
	ys = [-1,-1,-1,-1,1,1,1]

	posOnes = []
	i = 0
	while (i < len(datax1POS)):
		posOnes.append(1)
		i += 1
	negOnes = []
	i = 0
	while (i < len(datax1NEG)):
		negOnes.append(-1)
		i += 1

	dataPOS.append(posOnes)
	dataPOS.append(datax1POS)
	dataPOS.append(datax2POS)
	dataNEG.append(negOnes)
	dataNEG.append(datax1NEG)
	dataNEG.append(datax2NEG)

	return (dataPOS, dataNEG)

def getDistance(testPoint, dataPointx1, dataPointx2):
	return math.sqrt( ((testPoint[1]-dataPointx1)**2) + ((testPoint[2]-dataPointx2)**2) )

def getMin(testPoint, dataPOS, dataNEG):
	posMin = 65536
	negMin = 65536
	i = 0
	while (i < len(dataPOS[0])):
		testNum = getDistance(testPoint, dataPOS[1][i], dataPOS[2][i])
		if (testNum < posMin):
			posMin = testNum
		i += 1
	i = 0
	while (i < len(dataNEG[0])):
		testNum = getDistance(testPoint, dataNEG[1][i], dataNEG[2][i])
		if (testNum < negMin):
			negMin = testNum
		i += 1

	if (posMin <= negMin):
		return 1
	elif (posMin > negMin):
		return -1


def test(testPoint, dataPOS, dataNEG):
	result = getMin(testPoint, dataPOS, dataNEG)
	if (result == 1):
		plt.plot(testPoint[1], testPoint[2], 'bs', label = "testpoint")
	elif (result == -1):
		plt.plot(testPoint[1], testPoint[2], 'rs', label = "testpoint")


def plot(dataPOS, dataNEG):
	fig = plt.figure()
	plt.plot(dataPOS[1], dataPOS[2], 'bo', label = "1")
	plt.plot(dataNEG[1], dataNEG[2], 'rx', label = "-1")
	fig.suptitle('Nearest Neighbor', fontsize = 20)
	plt.xlabel('x1', fontsize = 18)
	plt.ylabel('x2', fontsize = 18)
	plt.legend(loc = 'upper right')
	plt.axis([-2.5,2.5,-2.5,2.5])

if __name__ == "__main__":
	dataPOS, dataNEG = getData()
	plot(dataPOS, dataNEG)
	test([1,-1,1], dataPOS, dataNEG)
	test([1,1,-2], dataPOS, dataNEG)
	plt.show()
