import numpy
import scipy
import matplotlib
import copy

import matplotlib.pyplot as plt
from random import randint
import random

w = [-1, 1, 1]

def formula(x):
	print("w: ", w)
	return ((-w[1]*x)/w[2])-(w[0]/w[2])

def formula2(x):
	return (x*wlinslope + wlinyint)

def cart2pol(x, y):
	rho = numpy.sqrt(x**2 + y**2)
	theta = numpy.arctan2(y, x)
	return[rho, theta]

def pol2cart(rho, theta):
	x = rho * numpy.cos(theta)
	y = rho * numpy.sin(theta)
	return[x, y]

def plot(data, h, type):
	i = 0
	rx = []
	ry = []
	bx = []
	by = []

	#print("data: ", data)

	while (i < len(data)):
		if (h[i] == -1):
			bx.append(data[i][0+type])
			by.append(data[i][1+type])
		elif (h[i] == 1):
			rx.append(data[i][0+type])
			ry.append(data[i][1+type])
		i += 1	

	if (type == 0):
		lx = numpy.linspace(0,upperBound,2000)

		gx = numpy.array(range(0, upperBound))
		gy = formula(gx)
		plt.plot(gx, gy)

		plt.plot(rx, ry, 'bo')
		plt.plot(bx, by, 'rx')
		plt.axis([0,upperBound,0,upperBound])
		plt.show()

def plotwlin(wlin):
	wlinslope = -1.0*wlin[1][0]/wlin[2][0]
	wlinyint = -1.0*wlin[0][0]/wlin[2][0]

	print("wlinslope: ", wlinslope)
	print("wlinyint: ", wlinyint)

	gx = numpy.array(range(0, upperBound))
	gy = formula2(gx)
	plt.plot(gx, gy)

def checkBad(k, data, h):
	insideSign = w[0]*1+w[1]*data[k][0]+w[2]*data[k][1]
	if (insideSign > 0 and h[k] > 0):
		return True
	elif (insideSign > 0 and h[k] < 0):
		return False
	elif (insideSign < 0 and h[k] < 0):
		return True
	elif (insideSign < 0 and h[k] > 0):
		return False

def updateWeights(k, data, h):
	#print("wpre: ", w)
	j = 0
	while (j < 3):
		if (j == 0):
			w[j] = w[j] + (h[k])
		else:
			w[j] = w[j] + (h[k]*data[k][j-1])
		j += 1
	#print("wafter: ", w)

def makeDataCircles(data):
	i = 0
	while (i < numPoints):
		blueorred = random.randrange(-1,2,2)
		if (blueorred == -1):
			theta = random.uniform(0,180)
			rho = random.uniform(10,15)
			location = pol2cart(rho, theta)
			location[0] += 17.5
			if (location[1] < 0):
				location[1] = -(location[1])
			location[1] += 27.5
			blue.append(location[0])
			blue.append(location[1])
			data.append(list(blue))
			blue.clear()
			h.append(-1)
		elif (blueorred == 1):
			theta = random.uniform(180,360)
			rho = random.uniform(10,15)
			location = pol2cart(rho, theta)
			location[0] += 30
			if (location[1] > 0):
				location[1] = -(location[1])
			location[1] += 22.5
			red.append(location[0])
			red.append(location[1])
			data.append(list(red))
			red.clear()
			h.append(1)
		i += 1
	return h

def fixData(data):
	data2 = copy.deepcopy(data)
	i = 0
	while (i < len(data2)):
		data2[i].insert(0, 1)
		i += 1

	return data2

def runSimulation(h ,data):
	numIter = 0
	complete = False
	while(complete == False): ##while iterations still need to be done
		i = 0
		checking = False
		while (i < len(data)): ##looping through data
			checking = checkBad(i, data, h)
			if (checking == True):
				if (i == len(data)-1): ##if you're done
					complete = True
					break
				else:
					i += 1
					continue
			elif (checking == False): ##if there is a wrong value
				updateWeights(i, data, h)
				break
			i += 1
		numIter += 1
	print("numIter: ", numIter)

if __name__ == "__main__":
	x1 = []
	x2 = []

	data = []
	datacircles = []
	red = []
	blue = []
	h = []
	h2 = []

	numPoints = 2000
	upperBound = 50

	h = makeDataCircles(datacircles)
	#print("data:", data)
	#print("h: ", h2)

	runSimulation(h, datacircles)
	#print("data:", data)

	#regression: 3.1b
	datacircles2 = fixData(datacircles)
	x_matrix = numpy.matrix(datacircles2)

	#h_fix = fixh(h)
	y_matrix = numpy.matrix(h)
	#print("x_matrix: ", x_matrix)
	#print("y_matrix: ", y_matrix)
	wlin = (numpy.linalg.inv(numpy.transpose(x_matrix) * x_matrix) * numpy.transpose(x_matrix)) * numpy.transpose(y_matrix)
	print("wlin: ", wlin)
	wlinslope = -1.0*wlin[1][0]/wlin[2][0]
	wlinyint = -1.0*wlin[0][0]/wlin[2][0]
	wlinslope = wlinslope.item(0,0)
	wlinyint = wlinyint.item(0,0)
	#w2 = 

	plotwlin(wlin)
	plot(datacircles, h, 0)
	#plot(data2, h, 1)
