import numpy
import scipy
import matplotlib

import matplotlib.pyplot as plt
from random import randint
import random

def getIntensity(trainingDigit):
	intensity = 0
	i = 0
	while (i < 256):
		intensity += (float(trainingDigit[i]) + 1)
		i += 1
	return (intensity/256)

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

def getData(xs, ys):
	databack = []
	singledata = []
	i = 0
	while (i < len(xs)):
		singledata = []
		singledata.append(1)
		singledata.append(xs[i])
		singledata.append(ys[i])
		databack.append(singledata)
		i += 1

	return databack

def getYs(x1, x5):
	ys = []
	i = 0
	while (i < len(x1)):
		ys.append(1)
		i += 1
	i = 0
	while (i < len(x5)):
		ys.append(-1)
		i += 1

	return ys

def plotwlin(wlin):
	wlinslope = -1.0*wlin[1][0]/wlin[2][0]
	wlinyint = -1.0*wlin[0][0]/wlin[2][0]

	print("wlinslope: ", wlinslope)
	print("wlinyint: ", wlinyint)

	gx = numpy.array(range(0, upperBound))
	gy = formula2(gx)
	plt.plot(gx, gy)


if __name__ == "__main__":
	f = open("ZipDigits.train", 'r')

	x1 = []
	y1 = []
	x5 = []
	y5 = []

	for line in f:
		line = line.split(' ')
		if (line[0] == '5.0000'):
			line = line[1:-1]
			x5.append(getSymmetry(line))
			y5.append(getIntensity(line))
		if (line[0] == '1.0000'):
			line = line[1:-1]
			x1.append(getSymmetry(line))
			y1.append(getIntensity(line))

	#make data with 1 in beginning
	data = []
	data.append(getData(x1, y1))
	data.append(getData(x5, y5))
	print("data: ", data)

	ys = getYs(x1, x5)

	#get wlin
	x_matrix = numpy.matrix(data)
	y_matrix = numpy.matrix(ys)
	print("x_matrix: ", x_matrix)
	print("y_matrix: ", y_matrix)
	wlin = (numpy.linalg.inv(numpy.transpose(x_matrix) * x_matrix) * numpy.transpose(x_matrix)) * numpy.transpose(y_matrix)
	print("wlin: ", wlin)
	wlinslope = -1.0*wlin[1][0]/wlin[2][0]
	wlinyint = -1.0*wlin[0][0]/wlin[2][0]
	wlinslope = wlinslope.item(0,0)
	wlinyint = wlinyint.item(0,0)
	plotwlin(wlin)

	#now, plot
	fig = plt.figure()
	plt.plot(x5, y5, 'rx')
	plt.plot(x1, y1, 'bo')
	fig.suptitle('digits', fontsize = 20)
	plt.xlabel('asymmetry', fontsize = 18)
	plt.ylabel('intensity', fontsize = 16)
	plt.show()