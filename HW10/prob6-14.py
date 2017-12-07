import numpy
import scipy
import matplotlib
import math

import matplotlib.pyplot as plt
from random import randint
import random

from PointHolder import *
from Point import *

def getDistance(point1, point2):
	return math.sqrt( ((point1.x1 - point2.x1)**2) + ((point1.x2 - point2.x2)**2) )

def getDataRandom(numPoints):
	pointData = PointHolder()
	i = 0
	while (i < numPoints):
		x1 = random.uniform(0,1)
		x2 = random.uniform(0,1)
		pointData.addPoint(Point(x1, x2))
		i += 1

	return pointData

def getDataUniform(numPoints):
	pointData = PointHolder()
	i = 0.
	while (i < 1):
		j = 0.
		while (j < 1):
			pointData.addPoint(Point(i, j))
			j += 0.1
		i += 0.1

	return pointData

def getCenters(points, numCenters, numPoints):
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

	i = 0
	while (i < len(groups)):
		print(groups[i].getLength())
		i += 1
		
	avgs = PointHolder()
	i = 0
	while (i < len(groups)):
		avgPoint = Point((groups[i]).getAvgx1(), (groups[i]).getAvgx2())
		avgs.addPoint(avgPoint)
		i += 1

	print(avgs)

def plot(points):
	i = 0
	while (i < points.getLength()):
		plt.plot((points.getPoint(i)).x1, (points.getPoint(i)).x2, 'bo')
		i += 1

def main():
	numPoints = 100
	numCenters = 10
	dataUniform = getDataUniform(numPoints)
	dataRandom = getDataRandom(numPoints)

	getCenters(dataUniform, numCenters, numPoints)
	
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
	plot(dataUniform)

	plt.show()

if __name__ == "__main__":
	main()