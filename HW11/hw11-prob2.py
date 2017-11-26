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

def getData(numPoints):
	pointData = PointHolder()
	i = 0
	while (i < numPoints):
		x1 = random.uniform(0,1)
		x2 = random.uniform(0,1)
		pointData.addPoint(Point(x1, x2))
		i += 1

	return pointData
	
def runLloyds(points, numCenters, numPoints):
	centerPoints = PointHolder()

	#get center
	randomCenter = points.getPoint(random.randint(0,numPoints))
	centerPoints.addPoint(randomCenter)
	i = 1
	while (i < numCenters):
		newCenter = points.getNewCenter(centerPoints)
		centerPoints.addPoint(newCenter)
		i += 1

	#assign points
	groups = [PointHolder(), PointHolder(), PointHolder(), PointHolder(), PointHolder(), PointHolder(), PointHolder(), PointHolder(), PointHolder(), PointHolder()]

	i = 0
	while (i < numPoints):
		point = points.getPoint(i)
		minDistance = 65536
		incNum = 0
		j = 0
		while (j < numCenters):
			distance = getDistance(point, centerPoints.getPoint(j))
			if (distance < minDistance):
				minDistance = distance
				incNum = j
			j += 1 

		groups[incNum].addPoint(point)
		i += 1

	# i = 0
	# while (i < len(groups)):
	# 	print(groups[i].getLength())
	# 	i += 1
	i = 0
	while (i < len(groups)):
		avgPoint = Point(groups[i].getAvg)
		i += 1


def plot(points):
	i = 0
	while (i < points.getLength()):
		plt.plot((points.getPoint(i)).x1, (points.getPoint(i)).x2, 'bo')
		i += 1

def main():
	numPoints = 200
	numCenters = 10
	data = getData(numPoints)

	runLloyds(data, numCenters, numPoints)
	
	x1beg = 0
	x1end = 1
	x2beg = 0
	x2end = 1

	fig = plt.figure()
	fig.suptitle('Clustering', fontsize = 20)
	plt.xlabel('x1', fontsize = 18)
	plt.ylabel('x2', fontsize = 18)
	plt.legend(loc = 'upper right')
	plt.axis([x1beg, x1end, x2beg, x2end])
	plot(data)

	plt.show()

if __name__ == "__main__":
	main()
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
	plt.axis([0, 100, 0, 0.075])
	plt.show()