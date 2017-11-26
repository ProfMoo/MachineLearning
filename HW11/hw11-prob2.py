import numpy
import scipy
import matplotlib
import math

import matplotlib.pyplot as plt
from random import randint
import random

from Point import *
from PointHolder import *

def normalizeFeatures(x1, xN, y1, yN):
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

	return (x1, xN, y1, yN)

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

def getData(x1, xN, y1, yN):
	pointData = PointHolder()
	i = 0
	while (i < len(x1)):
		pointData.addPoint(Point(x1[i], y1[i], 1))
		i += 1
	i = 0
	while (i < len(xN)):
		pointData.addPoint(Point(xN[i], yN[i], -1))
		i += 1

	return pointData

def runLloyds(points, numCenters, numPoints):
	centerPoints = PointHolder()

	#get center
	randomCenter = points.getPoint(random.randint(0,numPoints))
	print("rc: ", randomCenter.x1)
	centerPoints.addPoint(randomCenter)
	print("cplen: ", centerPoints.getLength())
	print("points: ", (points.getPoint(1)).x1)
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
		avgPoint = Point(groups[i].getAvgx1(), groups[i].getAvgx2(), 0)
		avgPointHolder = PointHolder()
		avgPointHolder.addPoint(avgPoint)
		plot(avgPointHolder, 1)
		i += 1

def plot(points, type):
	i = 0
	while (i < points.getLength()):
		if (type == 0):
			plt.plot((points.getPoint(i)).x1, (points.getPoint(i)).x2, 'bo')
		if (type == 1):
			plt.plot(float(points.getPoint(i).x1), float(points.getPoint(i).x2), 'ro')
		i += 1

def main(file):
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

	numPoints = 300
	numCenters = 10
	x1, xN, y1, yN = normalizeFeatures(x1, xN, y1, yN)

	data = getData(x1, xN, y1, yN)
	print('here: ', data.getPoint(2).x1)

	x1beg = -1
	x1end = 1
	x2beg = -1
	x2end = 1

	fig = plt.figure()
	fig.suptitle('Clustering', fontsize = 20)
	plt.xlabel('x1', fontsize = 18)
	plt.ylabel('x2', fontsize = 18)
	plt.legend(loc = 'upper right')
	plt.axis([x1beg, x1end, x2beg, x2end])


	runLloyds(data, numCenters, len(x1) + len(xN))

	plot(data, 0)

	plt.show()

if __name__ == "__main__":
	main("ZipDigits.train")