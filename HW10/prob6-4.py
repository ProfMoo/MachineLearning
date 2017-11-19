import numpy
import scipy
import matplotlib
import math

import matplotlib.pyplot as plt
from random import randint
import random

from NNOB import *

def cart2pol(x, y):
	rho = numpy.sqrt(x**2 + y**2)
	theta = numpy.arctan2(y, x)
	return[rho, theta]

def pol2cart(rho, theta):
	x = rho * numpy.cos(theta*math.pi)
	y = rho * numpy.sin(theta*math.pi)
	return(x, y)

def getData(numPoints):
	dataNEG = []
	dataPOS = []
	i = 0
	while (i < 3):
		dataPOS.append(list())
		dataNEG.append(list())
		i += 1

	i = 0	
	while (i < numPoints):
		blueorred = random.randrange(-1,2,2)
		if (blueorred == 1):
			theta = random.uniform(1,2)
			rho = random.uniform(10,15)
			x1, x2 = pol2cart(rho, theta)
			x1 += 30
			x2 += 22.5
			dataPOS[0].append(1)
			dataPOS[1].append(x1)
			dataPOS[2].append(x2)
		if (blueorred == -1):
			red = []
			theta = random.uniform(0,1)
			rho = random.uniform(10,15)
			x1, x2 = pol2cart(rho, theta)
			x1 += 17.5
			x2 += 27.5
			dataNEG[0].append(1)
			dataNEG[1].append(x1)
			dataNEG[2].append(x2)
		i += 1
	return (dataPOS, dataNEG)

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

def plotOG(dataPOS, dataNEG, x1beg, x1end, x2beg, x2end):
	plt.plot(dataPOS[1], dataPOS[2], 'bs', label = "1")
	plt.plot(dataNEG[1], dataNEG[2], 'rs', label = "-1")

def main():
	dataPOS, dataNEG = getData(500)
	x1beg = 0
	x1end = 50
	x2beg = 0
	x2end = 50
	increment = 1

	fig = plt.figure()
	fig.suptitle('Nearest Neighbor 3-1', fontsize = 20)
	plt.xlabel('x1', fontsize = 18)
	plt.ylabel('x2', fontsize = 18)
	plt.legend(loc = 'upper right')
	plt.axis([x1beg, x1end, x2beg, x2end])

	plotOG(dataPOS, dataNEG, x1beg, x1end, x2beg, x2end)
	NNPOS, NNNEG = makeGraph(3, x1beg, x1end, x2beg, x2end, increment, dataPOS, dataNEG)
	plot(NNPOS, NNNEG, x1beg, x1end, x2beg, x2end)

	plt.show()

if __name__ == "__main__":
	main()
