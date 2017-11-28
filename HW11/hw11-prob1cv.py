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
		# if (i%1 < 0.01):
		# 	print("i: ", i)
		j = startingX2
		while (j < endingX2):
			result = getMin(NN, [1,i,j], dataPOS, dataNEG)
			if (result == 1):
				NNdataPOS[0].append(1)
				NNdataPOS[1].append(float("{0:.2f}".format(i)))
				NNdataPOS[2].append(float("{0:.2f}".format(j)))
			if (result == -1):
				NNdataNEG[0].append(1)
				NNdataNEG[1].append(float("{0:.2f}".format(i)))
				NNdataNEG[2].append(float("{0:.2f}".format(j)))
			j += increment
		i += increment

	return (NNdataPOS, NNdataNEG)

def plot(dataPOS, dataNEG):
	plt.plot(dataPOS[1], dataPOS[2], color = "#99CCFF", linestyle = 'none', marker = 'x', label = "1 Data")
	plt.plot(dataNEG[1], dataNEG[2], color = "#FF9999", linestyle = 'none', marker = 'x', label = "-1 Data")

def plotOG(dataPOS, dataNEG):
	plt.plot(dataPOS[1], dataPOS[2], 'bo', label = "1 NN")
	plt.plot(dataNEG[1], dataNEG[2], 'rx', label = "-1 NN")

def roundDown(n, d=1):
	d = int('1' + ('0' * d))
	return math.floor(n * d)/d

def roundUp(n, d=1):
	d = int('1' + ('0' * d))
	return math.ceil(n * d)/d

def testPoint(data, i, j, trainingNNPOScv, trainingNNNEGcv):
	values = [0,0,0,0]
	locations = [[0,0],[0,0],[0,0],[0,0]]

	#bottomleft
	try:
		locations[0][0] = trainingNNPOScv[1][trainingNNPOScv[1].index(roundDown(data[i][1][j]))]
		locations[0][1] = trainingNNPOScv[2][trainingNNPOScv[2].index(roundDown(data[i][2][j]))]
		values[0] = 1
	except:
		locations[0][0] = trainingNNNEGcv[1][trainingNNNEGcv[1].index(roundDown(data[i][1][j]))]
		locations[0][1] = trainingNNNEGcv[2][trainingNNNEGcv[2].index(roundDown(data[i][2][j]))]
		values[0] = -1

	#bottomright
	try:
		locations[1][0] = trainingNNPOScv[1][trainingNNPOScv[1].index(roundUp(data[i][1][j]))]
		locations[1][1] = trainingNNPOScv[2][trainingNNPOScv[2].index(roundDown(data[i][2][j]))]
		values[1] = 1
	except:
		locations[1][0] = trainingNNNEGcv[1][trainingNNNEGcv[1].index(roundUp(data[i][1][j]))]
		locations[1][1] = trainingNNNEGcv[2][trainingNNNEGcv[2].index(roundDown(data[i][2][j]))]
		values[1] = -1

	#topleft
	try:
		locations[2][0] = trainingNNPOScv[1][trainingNNPOScv[1].index(roundUp(data[i][1][j]))]
		locations[2][1] = trainingNNPOScv[2][trainingNNPOScv[2].index(roundDown(data[i][2][j]))]
		values[2] = 1
	except:
		locations[2][0] = trainingNNNEGcv[1][trainingNNNEGcv[1].index(roundUp(data[i][1][j]))]
		locations[2][1] = trainingNNNEGcv[2][trainingNNNEGcv[2].index(roundDown(data[i][2][j]))]
		values[2] = -1

	#topright
	try:
		locations[3][0] = trainingNNPOScv[1][trainingNNPOScv[1].index(roundUp(data[i][1][j]))]
		locations[3][1] = trainingNNPOScv[2][trainingNNPOScv[2].index(roundUp(data[i][2][j]))]
		values[3] = 1
	except:
		locations[3][0] = trainingNNNEGcv[1][trainingNNNEGcv[1].index(roundUp(data[i][1][j]))]
		locations[3][1] = trainingNNNEGcv[2][trainingNNNEGcv[2].index(roundUp(data[i][2][j]))]
		values[3] = -1

	k = 0
	minDistance = 65536
	index = 65536
	while (k < 3):
		checkDis = getDistance([1, data[i][1][j], data[i][2][j]], locations[k][0], locations[k][1])
		if (checkDis < minDistance):
			index = k
			minDistance = checkDis
		k += 1

	# if (j < 2):
	# 	print("values: ", values)
	# 	print("locations: ", locations)

	if (values[index] == -1 and i == 0):
		return 1
	if (values[index] == 1 and i == 1):
		return 1
	else:
		return 0

def ETest(dataPOS, dataNEG):
	#USE trainingNNPOS, trainingNNNEG
	totalError = 0
	data = [dataPOS, dataNEG]

	i = 0
	while (i < len(data)):
		j = 0
		while (j < len(data[i][0])):
			totalError += testPoint(data, i, j, trainingNNPOS, trainingNNNEG)
			j += 1
		i += 1

	return totalError/(len(dataPOS[0]) + len(dataNEG[0]))

def cv(dataPOS, dataNEG, hs, ktop, x1beg, x1end, x2beg, x2end, increment):
	klist = []
	cvlist = []
	i = 0
	# while (i < ktop/2):
	while (i < 50):
		print("i: ", i)
		j = 0
		numWrong = 0
		while (j < len(dataPOS[0])/2):

			if (10 < j < 15):
				print("j: ", j)

			#make new list
			newList = []
			k = 0
			while (k < 3):
				newList.append(list())
				k += 1
			k = 0
			while (k < j):
				newList[0].append(dataPOS[0][k])
				newList[1].append(dataPOS[1][k])
				newList[2].append(dataPOS[2][k])
				k += 1
			k = (j+1)
			while (k < len(dataPOS[0])):
				newList[0].append(dataPOS[0][k])
				newList[1].append(dataPOS[1][k])
				newList[2].append(dataPOS[2][k])
				k += 1

			#make new NN
			cvPOS, cvNEG = makeGraph(i, x1beg, x1end, x2beg, x2end, increment, newList, dataNEG)

			#get error
			if (testPoint([dataPOS, dataNEG], 0, j, cvPOS, cvNEG) == 1):
				numWrong += 1
			j += 1

		j = 0
		numWrong = 0
		while (j < len(dataNEG[0])/2):

			if (10 < j < 15):
				print("j: ", j)

			#make new list
			newList = []
			k = 0
			while (k < 3):
				newList.append(list())
				k += 1
			k = 0
			while (k < j):
				newList[0].append(dataNEG[0][k])
				newList[1].append(dataNEG[1][k])
				newList[2].append(dataNEG[2][k])
				k += 1
			k = (j+1)
			while (k < len(dataNEG[0])):
				newList[0].append(dataNEG[0][k])
				newList[1].append(dataNEG[1][k])
				newList[2].append(dataNEG[2][k])
				k += 1

			#make new NN
			cvPOS, cvNEG = makeGraph(i, x1beg, x1end, x2beg, x2end, increment, dataPOS, newList)

			#get error
			if (testPoint([dataPOS, dataNEG], 1, j, cvPOS, cvNEG) == 1):
				numWrong += 1
			j += 1

		klist.append(i)
		cvlist.append(numWrong/(len(dataPOS[0]) + len(dataNEG[0])))
		i += 1

	print("klist: ", klist)
	print("cvlist: ", cvlist)
	return (klist, cvlist)

def handleTrain(file):
	f = open(file, 'r')

	x1 = []
	y1 = []
	xN = []
	yN = []
	increment = 0.10

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

	kList, CVAnswer = cv(dataPOS, dataNEG, hs, 300, x1beg, x1end, x2beg, x2end, increment)

	fig = plt.figure()
	fig.suptitle('Nearest Neighbor', fontsize = 20)
	plt.xlabel('x1', fontsize = 18)
	plt.ylabel('x2', fontsize = 18)
	plt.legend(loc = 'upper right')
	plt.axis([x1beg, x1end, x2beg, x2end])

	#part 1
	NNPOS = 0
	NNNEG = 0
	#NNPOS, NNNEG = makeGraph(5, x1beg, x1end, x2beg, x2end, increment, dataPOS, dataNEG)
	#NNPOS, NNNEG = makeGraph(5, x1beg, x1end, x2beg, x2end, 0.01, dataPOS, dataNEG)
	#plot(NNPOS, NNNEG)
	#plotOG(dataPOS, dataNEG)
	#plt.show()

	return (NNPOS, NNNEG, hs, kList, CVAnswer)

def handleTest(file):
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

	error = ETest(dataPOS, dataNEG)
	print("error: ", error)

	#plotOG(dataPOS, dataNEG)
	#plt.show()

if __name__ == "__main__":
	trainingNNPOS, trainingNNNEG, trainingHs, kList, CVAnswer = handleTrain("ZipDigits.train")
	#plt.clf()
	#handleTest("ZipDigits.test")

	fig = plt.figure()
	plt.plot(kList, CVAnswer, 'ro', label='cv')
	#plt.plot(LambdaList, EtestAnswer, 'bo', label='etest')
	fig.suptitle('CV calculation', fontsize = 20)
	plt.xlabel('k', fontsize = 18)
	plt.ylabel('error', fontsize = 18)
	plt.legend(loc = 'upper right')
	plt.axis([0, 100, 0, 0.5])
	plt.show()

'''
klist:  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
cvlist:  [0.42, 0.09666666666666666, 0.13666666666666666, 0.013333333333333334, 0.04, 0.013333333333333334, 0.013333333333333334, 0.013333333333333334, 0.013333333333333334, 0.013333333333333334, 0.013333333333333334, 0.013333333333333334, 0.013333333333333334, 0.013333333333333334, 0.013333333333333334, 0.013333333333333334, 0.013333333333333334, 0.013333333333333334, 0.013333333333333334, 0.013333333333333334, 0.013333333333333334, 0.013333333333333334, 0.013333333333333334, 0.013333333333333334, 0.013333333333333334, 0.013333333333333334, 0.013333333333333334, 0.013333333333333334, 0.013333333333333334, 0.013333333333333334, 0.013333333333333334, 0.013333333333333334, 0.013333333333333334, 0.013333333333333334, 0.013333333333333334, 0.013333333333333334, 0.013333333333333334, 0.013333333333333334, 0.013333333333333334, 0.013333333333333334, 0.013333333333333334, 0.013333333333333334, 0.013333333333333334, 0.013333333333333334, 0.013333333333333334, 0.013333333333333334, 0.013333333333333334, 0.013333333333333334, 0.013333333333333334, 0.013333333333333334]
'''