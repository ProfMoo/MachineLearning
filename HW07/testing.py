import numpy
import scipy
import matplotlib

import matplotlib.pyplot as plt
from random import randint
import random

w = [-1,1,1]

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

def getData(x1, y1, x5, y5, one):
	databack = []
	singledata = []
	i = 0
	while (i < len(x1)):
		singledata = []
		if (one == 1):
			singledata.append(1)
		singledata.append(x1[i])
		singledata.append(y1[i])
		databack.append(singledata)
		i += 1

	i = 0
	while (i < len(x5)):
		singledata = []
		if (one == 1):
			singledata.append(1)
		singledata.append(x5[i])
		singledata.append(y5[i])
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

def formula2(x, wslope, wyint):
	return (x*wslope + wyint)

def plotwlin(wother, which, whichFile):

	print("wother:", wother)

	'''
		used for testing data
		gx = numpy.array(range(105, 113))
		gx = numpy.array(range(90, 97))
	'''


	if (which == 0):
		wslope = -1.0*wother[1][0]/wother[2][0]
		wyint = -1.0*wother[0][0]/wother[2][0]
		wslope = wslope.item(0,0)
		wyint = wyint.item(0,0)
		print("wslope: ", wslope)
		print("wyint: ", wyint)
		if (whichFile == "test"):
			gx = numpy.array(range(105, 113))
		if (whichFile == "train"):
			gx = numpy.array(range(97, 108))
		gy = formula2(gx, wslope, wyint)
		#plt.plot(gx, gy, 'r')
	if (which == 1):
		wslope = -1.0*wother[1]/wother[2]
		wyint = -1.0*wother[0]/wother[2]
		print("wslope: ", wslope)
		print("wyint: ", wyint)
		if (whichFile == "test"):
			gx = numpy.array(range(94, 103))
		if (whichFile == "train"):
			gx = numpy.array(range(95, 108))
		gy = formula2(gx, wslope, wyint)
		plt.plot(gx, gy, 'b')

def checkBad(k, data, h, wcheck):
	insideSign = wcheck[0]*1+wcheck[1]*data[k][0]+wcheck[2]*data[k][1]
	if (insideSign > 0 and h[k] > 0):
		return True
	elif (insideSign > 0 and h[k] < 0):
		return False
	elif (insideSign < 0 and h[k] < 0):
		return True
	elif (insideSign < 0 and h[k] > 0):
		return False

def pocketCheck(wbest, data, h):
	i = 0
	wbestbad = 0
	wbad = 0
	while (i < len(data)):
		if (checkBad(i, data, h, w) == False):
			wbad += 1
		if (checkBad(i, data, h, wbest) == False):
			wbestbad += 1
		i += 1

	if (wbad < wbestbad):
		print("wbad: ", wbad)
		print("wbestbad: ", wbestbad)
		print("found better, replacing...")
		print("w: ", w)
		return True
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

def runSimulation(h, data):
	numIter = 0
	complete = False
	wbest = w[:]
	print("wbest(wlin right now): ", wbest)
	while(complete == False and numIter < pocketTop): ##while iterations still need to be done
		
		if (numIter%100 == 0):
			print("numIter: ", numIter)

		#check for pocket algorithm
		if(pocketCheck(wbest, data, h)):
			wbest = w[:]

		i = 0
		checking = False
		while (i < len(data)): ##looping through data
			checking = checkBad(i, data, h, w)
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

	return wbest

def EinCalc(data, h, wtotest):
	i = 0
	wtotestbad = 0
	while (i < len(data)):
		if (checkBad(i, data, h, wtotest) == False):
			wtotestbad += 1
		i += 1

	return (wtotestbad/len(data))

if __name__ == "__main__":
	f = open("ZipDigits.test", 'r')
	whichFile = "test"

	pocketTop = 1000
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
	data2 = []
	data = getData(x1, y1, x5, y5, 1)
	data2 = getData(x1, y1, x5, y5, 0)
	#print("data: ", data)
	ys = getYs(x1, x5)

	#get wlin
	# x_matrix = numpy.matrix(data)
	# y_matrix = numpy.matrix(ys)
	# print("x_matrix: ", x_matrix)
	# print("y_matrix: ", y_matrix)
	# x_matrix_trans = numpy.transpose(x_matrix)
	# wlin = (numpy.linalg.inv(x_matrix_trans * x_matrix) * x_matrix_trans) * numpy.transpose(y_matrix)
	# print("wlin: ", wlin)

	# w[0] = wlin[0].item(0,0)
	# w[1] = wlin[1].item(0,0)
	# w[2] = wlin[2].item(0,0)

	# wbest = runSimulation(ys, data2)

	wbest = [320.645, -3.384, 34.6975]
	#now, plot
	fig = plt.figure()
	plt.plot(x5, y5, 'rx')
	plt.plot(x1, y1, 'bo')

	#calculate Ein
	print("wbest: ", wbest)
	Ein = EinCalc(data2, ys, wbest)
	print("Ein: ", Ein)
	
	plotwlin(wbest, 1, whichFile)

	fig.suptitle('digits', fontsize = 20)
	plt.xlabel('asymmetry', fontsize = 18)
	plt.ylabel('intensity', fontsize = 16)
	plt.show()