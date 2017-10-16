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
			pointOne = float(trainingDigit[j*16+1])
			pointTwo = float(trainingDigit[(j+1)*16 - (i + 1)])
			xsymmetry += abs(pointOne - pointTwo)
			j += 1
		i += 1
	i = 0
	while (i < 16): #get symmetry across horizontal axis
		j = 0
		while (j < 8):
			pointOne = float(trainingDigit[j*16+1])
			pointTwo = float(trainingDigit[(15-j)*16-i])
			ysymmetry += abs(pointOne - pointTwo)
			j += 1
		i += 1
	return (xsymmetry+ysymmetry)/256.

# def getData(data, trainingDigits):
# 	i = 0
# 	while (i < len(trainingDigits)):
# 		singleData = []
# 		symmetry = getSymmetry(trainingDigits[i])
# 		singleData.append(symmetry)
# 		intensity = getIntensity(trainingDigits[i])
# 		singleData.append(intensity)
# 		data.append(singleData)
# 		i += 1

if __name__ == "__main__":
	f = open("ZipDigits.test", 'r')

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

	#now, plot
	fig = plt.figure()
	plt.plot(x5, y5, 'rx')
	plt.plot(x1, y1, 'bo')
	fig.suptitle('digits', fontsize = 20)
	plt.xlabel('asymmetry', fontsize = 18)
	plt.ylabel('intensity', fontsize = 16)
	plt.show()