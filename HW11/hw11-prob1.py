import numpy
import scipy
import matplotlib
import math

import matplotlib.pyplot as plt
from random import randint
import random

from NNOB import *

def normalizeFeatures(x1, y1, xN, yN):
	minX = 10000
	maxX = -10000
	minY = 10000
	maxY = -10000

	i = 0
	while (i < len(x1)):
		if (x1[i] < minX):
			minX = x1[i]
		if (x1[i] > maxX):
			maxX = x1[i]
		if (y1[i] < minY):
			minY = y1[i]
		if (y1[i] > maxY):
			maxY = y1[i]
		i += 1
	i = 0
	while (i < len(xN)):
		if (xN[i] < minX):
			minX = xN[i]
		if (xN[i] > maxX):
			maxX = xN[i]
		if (yN[i] < minY):
			minY = yN[i]
		if (yN[i] > maxY):
			maxY = yN[i]
		i += 1

	#now, subtract by (half the distance (from min to max) plus min value)
	valueToSubX = ((maxX - minX)/2) + minX 
	valueToSubY = ((maxY - minY)/2) + minY
	i = 0
	while (i < len(x1)):
		x1[i] = (x1[i]-valueToSubX)/(maxX/2)
		y1[i] = (y1[i]-valueToSubY)/(maxY/2)
		i += 1
	i = 0
	while (i < len(xN)):
		xN[i] = (xN[i]-valueToSubX)/(maxX/2)
		yN[i] = (yN[i]-valueToSubY)/(maxY/2)
		i += 1 

	return (x1, y1, xN, yN)

def getIntensity(trainingDigit):
	intensity = 0
	i = 0
	while (i < 256):
		intensity += (float(trainingDigit[i]) + 1)
		i += 1
	return (intensity)

def getSymmetry(trainingDigit):
	xsymmetry = 0
	ysymmetry = 0
	i = 0
	while (i < 8): #get symmetry across vertical axis
		j = 0
		while (j < 16):
			pointOne = float(trainingDigit[j * 16 + i])
			pointTwo = float(trainingDigit[(j + 1) * 16 - (i + 1)])
			xsymmetry += abs(pointOne - pointTwo)
			j += 1
		i += 1
	i = 0
	while (i < 16): #get symmetry across horizontal axis
		j = 0
		while (j < 8):
			pointOne = float(trainingDigit[j * 16 + i])
			pointTwo = float(trainingDigit[(15 - j) * 16 - i])
			ysymmetry += abs(pointOne - pointTwo)
			j += 1
		i += 1
	return (xsymmetry+ysymmetry)

def getData(x1, y1, xN, yN):
	dataPOS = []
	dataNEG = []

	i = 0
	while (i < 3):
		dataPOS.append(list())
		dataNEG.append(list())
		i += 1

	i = 0
	while (i < len(x1)):
		dataPOS[0].append(1)
		dataPOS[1].append(x1[i])
		dataPOS[2].append(y1[i])
		i += 1

	i = 0
	while (i < len(xN)):
		dataNEG[0].append(1)
		dataNEG[1].append(xN[i])
		dataNEG[2].append(yN[i])
		i += 1

	return (dataPOS, dataNEG)

def getHs(x1, xN):
	hs = []
	i = 0
	while (i < len(x1)):
		hs.append(1)
		i += 1
	i = 0
	while (i < len(xN)):
		hs.append(-1)
		i += 1

	return hs

def getDistance(testPoint, dataPointx1, dataPointx2):
	return math.sqrt( ((testPoint[1]-dataPointx1)**2) + ((testPoint[2]-dataPointx2)**2) )

def getMin(NN, testPoint, dataPOS, dataNEG):
	
	mins = []
	i = 0
	while (i < NN):
		mins.append(NNOB(65536, 0))
		i += 1

	i = 0
	while (i < len(dataPOS[0])):
		testNum = NNOB(getDistance(testPoint, dataPOS[1][i], dataPOS[2][i]), 1)

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
		if (i%1 < 0.01):
			print("i: ", i)
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
	plt.plot(dataPOS[1], dataPOS[2], color = "#99CCFF", linestyle = 'none', marker = 'x', label = "1 Data")
	plt.plot(dataNEG[1], dataNEG[2], color = "#FF9999", linestyle = 'none', marker = 'x', label = "-1 Data")

def plotOG(dataPOS, dataNEG, x1beg, x1end, x2beg, x2end):
	plt.plot(dataPOS[1], dataPOS[2], 'bo', label = "1 NN")
	plt.plot(dataNEG[1], dataNEG[2], 'rx', label = "-1 NN")
	# i = 0
	# while (i < len(dataPOS)):
	# 	plt.plot(dataPOS[i][1], dataPOS[i][2], 'bo', label = "1 NN")
	# 	i += 1
	# i = 0
	# while (i < len(dataNEG)):
	# 	plt.plot(dataNEG[i][1], dataNEG[i][2], 'rx', label = "-1 NN")
	# 	i += 1

def handle(file):
	f = open(file, 'r')

	x1 = []
	y1 = []
	xN = []
	yN = []

	for line in f:
		line = line.split(' ')
		if (line[0] != '1.0000'):
			line = line[1:-1]
			xN.append(getSymmetry(line))
			yN.append(getIntensity(line))
		if (line[0] == '1.0000'):
			line = line[1:-1]
			x1.append(getSymmetry(line))
			y1.append(getIntensity(line))

	print("lens: ", len(x1), len(xN))

	#normalize features
	x1, y1, xN, yN = normalizeFeatures(x1, y1, xN, yN)
	dataPOS, dataNEG = getData(x1, y1, xN, yN)

	print("len(data): ", len(dataPOS[0]), len(dataNEG[0]))

	hs = getHs(x1, xN)
	x1beg = -1.2
	x1end = 1.2
	x2beg = -1.2
	x2end = 1.2

	fig = plt.figure()
	fig.suptitle('Nearest Neighbor', fontsize = 20)
	plt.xlabel('x1', fontsize = 18)
	plt.ylabel('x2', fontsize = 18)
	plt.legend(loc = 'upper right')
	plt.axis([x1beg, x1end, x2beg, x2end])

	#part 1
	NNPOS, NNNEG = makeGraph(5, x1beg, x1end, x2beg, x2end, 0.01, dataPOS, dataNEG)
	plot(NNPOS, NNNEG, x1beg, x1end, x2beg, x2end)
	plotOG(dataPOS, dataNEG, x1beg, x1end, x2beg, x2end)

	plt.show()

if __name__ == "__main__":
	handle("ZipDigits.train")
