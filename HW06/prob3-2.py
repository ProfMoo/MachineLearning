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

		plt.plot(rx, ry, 'ro')
		plt.plot(bx, by, 'bx')
		plt.axis([0,upperBound,0,upperBound])
		plt.show()

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

def makeDataCircles(data, sep):
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
			location[1] += (25 + (sep/2.))
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
			location[1] += (25 - (sep/2.))
			red.append(location[0])
			red.append(location[1])
			data.append(list(red))
			red.clear()
			h.append(1)
		i += 1
	return h

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
	return numIter

if __name__ == "__main__":
	x1 = []
	x2 = []

	datacircles = []
	red = []
	blue = []
	h = []

	numPoints = 200
	upperBound = 50

	sepList = []
	numIters = []

	i = 0.2
	while (i < 5.1):
		w = [-1, 1, 1]
		i = round(i, 1)
		sepList.append(i)
		print("sepList: ", sepList)

		datacircles = []
		h = []
		h = makeDataCircles(datacircles, i)

		numIter = runSimulation(h, datacircles)
		#plot(datacircles, h, 0)
		numIters.append(numIter)

		print("numIters: ", numIters)
		i += 0.2
	
	fig = plt.figure()
	plt.plot(sepList, numIters, 'ro')
	print("numIters: ", numIters)
	print("sepList: ", sepList)
	fig.suptitle('3.2', fontsize = 20)
	plt.xlabel('sep', fontsize = 18)
	plt.ylabel('interations', fontsize = 16)
	plt.show()
	
	#print("data:", data)

	#plot(datacircles, h, 0)
	#plot(data2, h, 1)
