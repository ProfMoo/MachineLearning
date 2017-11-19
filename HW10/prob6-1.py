import numpy
import scipy
import matplotlib
import math

import matplotlib.pyplot as plt
from random import randint
import random

from NNOB import *

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

def tranOne(x1, x2):
	return math.sqrt(x1**2 + x2**2)

def tranTwo(x1, x2):
	return math.atan2(x2, x1)
	# if (x1 == 0):
	# 	if (x2 > 0):
	# 		return (math.pi/2)
	# 	elif (x2 < 0):
	# 		return (-math.pi/2)
	# return numpy.arctan(x2/x1)

def zTransform(dataPOS, dataNEG):
	zDataPOS = []
	zDataNEG = []
	i = 0
	while (i < 3):
		zDataPOS.append(list())
		zDataNEG.append(list())
		i += 1

	i = 0
	while (i < len(dataPOS[0])):
		zDataPOS[0].append(1)
		zDataPOS[1].append(tranOne(dataPOS[1][i], dataPOS[2][i]))
		zDataPOS[2].append(tranTwo(dataPOS[1][i], dataPOS[2][i]))
		i += 1
	i = 0
	while (i < len(dataNEG[0])):
		zDataNEG[0].append(1)
		zDataNEG[1].append(tranOne(dataNEG[1][i], dataNEG[2][i]))
		zDataNEG[2].append(tranTwo(dataNEG[1][i], dataNEG[2][i]))
		i += 1

	print("zDataPOS: ", zDataPOS)
	print("zDataNEG: ", zDataNEG)
	return (zDataPOS, zDataNEG)

def getDistance(testPoint, dataPointx1, dataPointx2):
	return math.sqrt( ((testPoint[1]-dataPointx1)**2) + ((testPoint[2]-dataPointx2)**2) )

def getMin(NN, testPoint, dataPOS, dataNEG):
	
	#make this generic
	mins = []
	i = 0
	while (i < NN):
		mins.append(NNOB(65536, 0))
		i += 1

	i = 0
	while (i < len(dataPOS[0])):
		testNum = NNOB(getDistance(testPoint, dataPOS[1][i], dataPOS[2][i]), 1)

		#looping is bad. need to replace biggest, not smallest
		j = (NN-1)
		while (j >= 0):
			if (testNum < mins[j]):
				mins[j] = testNum
				mins.sort()
				break
			j -= 1
		i += 1
	i = 0
	while (i < len(dataNEG[0])):
		testNum = NNOB(getDistance(testPoint, dataNEG[1][i], dataNEG[2][i]), -1)
		j = (NN-1)
		while (j >= 0):
			if (testNum < mins[j]):
				mins[j] = testNum
				mins.sort()
				break
			j -= 1
		i += 1

	i = 0
	numSum = 0
	while (i < NN):
		numSum += mins[i].classification
		i += 1

	if (numSum >= 0):
		return 1
	elif (numSum < 0):
		return -1

def makeGraph(NN, startingX1, endingX1, startingX2, endingX2, increment, dataPOS, dataNEG):
	NNdataPOS = []
	NNdataNEG = []

	#making place holder for data
	i = 0
	while (i < 3):
		NNdataPOS.append([])
		NNdataNEG.append([])
		i += 1

	#looping through all locations in NN graph
	i = startingX1
	while (i < endingX1):
		print("i:", i)
		j = startingX2
		while (j < endingX2):
			result = getMin(NN, [1,i,j], dataPOS, dataNEG)
			if (result == 1):
				NNdataPOS[0].append(1)
				NNdataPOS[1].append(i)
				NNdataPOS[2].append(j)
			if (result == -1):
				NNdataNEG[0].append(1)
				NNdataNEG[1].append(i)
				NNdataNEG[2].append(j)
			j += increment
		i += increment

	return (NNdataPOS, NNdataNEG)

def plot(dataPOS, dataNEG, x1beg, x1end, x2beg, x2end):
	plt.plot(dataPOS[1], dataPOS[2], 'bo', label = "1")
	plt.plot(dataNEG[1], dataNEG[2], 'rx', label = "-1")

def main():
	dataPOS, dataNEG = getData()
	zDataPOS, zDataNEG = zTransform(dataPOS, dataNEG)
	x1beg = -3.0
	x1end = 4.0
	x2beg = -3.0
	x2end = 5.0

	fig = plt.figure()
	fig.suptitle('Nearest Neighbor', fontsize = 20)
	plt.xlabel('x1', fontsize = 18)
	plt.ylabel('x2', fontsize = 18)
	plt.legend(loc = 'upper right')
	plt.axis([x1beg, x1end, x2beg, x2end])

	#part 1
	#plot(dataPOS, dataNEG, x1beg, x1end, x2beg, x2end)
	#NNPOS, NNNEG = makeGraph(1, x1beg, x1end, x2beg, x2end, 0.1, dataPOS, dataNEG)
	#plot(NNPOS, NNNEG, x1beg, x1end, x2beg, x2end)

	#plot(zDataPOS, zDataNEG, x1beg, x1end, x2beg, x2end)
	NNPOS, NNNEG = makeGraph(3, x1beg, x1end, x2beg, x2end, 0.1, zDataPOS, zDataNEG)
	plot(NNPOS, NNNEG, x1beg, x1end, x2beg, x2end)
	plt.show()

if __name__ == "__main__":
	main()
