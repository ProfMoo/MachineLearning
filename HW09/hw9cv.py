import numpy
import scipy
import matplotlib

import matplotlib.pyplot as plt
from random import randint
import random

features = 45
pocketTop = 500

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

def legendreData(x1, y1, degree):
	singlePoint = []

	i = 0
	while (i < (degree+1)):
		j = 0
		while (j < (i+1)):
			singlePoint.append(lp(i-j, x1) * lp(j, y1))
			j += 1
		i += 1

	return singlePoint

def getData8L(x1, y1, xN, yN):
	databack = []
	singledata = []
	polynum = 8

	i = 0
	while (i < len(x1)):
		singledata = []
		singledata = legendreData(x1[i], y1[i], polynum)
		databack.append(singledata)
		i += 1

	i = 0
	while (i < len(xN)):
		singledata = []
		singledata = legendreData(xN[i], yN[i], polynum)
		databack.append(singledata)
		i += 1

	return databack

def getYs(x1, xN):
	ys = []
	i = 0
	while (i < len(x1)):
		ys.append(1)
		i += 1
	i = 0
	while (i < len(xN)):
		ys.append(-1)
		i += 1

	return ys

def checkBad(k, data, h, wcheck):
	i = 0
	insideSign = 0
	while (i < len(wcheck)):
		insideSign += wcheck[i]*data[k][i]
		i += 1

	if (insideSign > 0 and h[k] > 0):
		return True
	elif (insideSign > 0 and h[k] < 0):
		return False
	elif (insideSign < 0 and h[k] < 0):
		return True
	elif (insideSign < 0 and h[k] > 0):
		return False

def getWReg(data, ys, lambdareg):
	x_matrix = numpy.matrix(data)
	y_matrix = numpy.matrix(ys)
	N = len(data[0])

	#print("x_matrix: ", x_matrix)
	#print("y_matrix: ", y_matrix)
	x_matrix_trans = numpy.transpose(x_matrix)
	regulation = lambdareg*numpy.identity(N)
	wreg = numpy.linalg.inv(x_matrix_trans * x_matrix + regulation) * x_matrix_trans * numpy.transpose(y_matrix)

	return wreg

def lp(k, x):
	if (k == 0):
		return 1
	if (k == 1):
		return x
	firstTerm = ((2*k - 1)/k)*x*lp(k - 1, x)
	secondTerm = ((k - 1)/k)*lp(k - 2, x)
	return firstTerm - secondTerm 

def cv(dataCV, ysCV, lambdatop): #cross validation calculator (from lambda 0 to 2)
	lambdaList = []
	ECVlist = []
	singleCvValue = 0

	i = 0
	while (i < lambdatop):
		print("CVlambda = ", i)
		numWrong = 0
		lambdaList.append(i)
		j = 0
		while (j < len(dataCV)):
			#make new list of data without a point
			newList = []
			k = 0
			while (k < j):
				newList.append(dataCV[k])
				k += 1
			k = (j+1)
			while (k < len(dataCV)):
				newList.append(dataCV[k])
				k += 1

			#make new list of ys without a point
			newYs = []
			k = 0
			while (k < j):
				newYs.append(ysCV[k])
				k += 1
			k = (j+1)
			while (k < len(ysCV)):
				newYs.append(ysCV[k])
				k += 1

			#make new Wreg(with lambda) and then check the point you left out
			wCV = getWReg(newList, newYs, i)
			if (checkBad(j, dataCV, ysCV, wCV) == False):
				numWrong += 1
			j += 1

		#append the number of points incorrect
		ECVlist.append(numWrong/(len(dataCV)))
		i += 0.1

	return (lambdaList, ECVlist)

def EinCalc(data, h, wtotest):
	i = 0
	wtotestbad = 0
	while (i < len(data)):
		if (checkBad(i, data, h, wtotest) == False):
			# print("BAD")
			wtotestbad += 1
		i += 1

	return (wtotestbad/len(data))

#needs to be changes to be on test set (currently ein on training)
def Etest(dataET, ysET, lambdatop):
	lambdaList = []
	EtestList = []
	i = 0
	while (i < lambdatop):
		print("Etestlambda: ", i)
		lambdaList.append(i)
		wET = getWReg(trainingData, trainingYs, i)
		#need to send in wET of training, data and ys of testing 
		EtestList.append(EinCalc(dataET, ysET, wET))
		i += 0.1

	return (EtestList)

def handle(file):
	lambdatop = 1.1
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

	#normalize features
	x1, y1, xN, yN = normalizeFeatures(x1, y1, xN, yN)

	#make data with 1 in beginning
	data = []
	data = getData8L(x1, y1, xN, yN)
	ys = getYs(x1, xN)
	
	if (file == "ZipDigits.train"):
		#updating global variables
		trainingData = data
		trainingYs = ys
		#calculate CV
		LambdaList, CVAnswer = cv(data, ys, lambdatop)
		return (LambdaList, CVAnswer)
	if (file == "ZipDigits.test"):
		#calculate Etest
		EtestAnswer = Etest(data, ys, lambdatop)
		return EtestAnswer

if __name__ == "__main__":
	trainingData = []
	trainingYs = []
	testingData = []
	testingYs = []

	LambdaList, CVAnswer = handle("ZipDigits.train")
	EtestAnswer = handle("ZipDigits.test")

	print("Lambda List: ", lambdaList)
	print("CVAnswer: ", CVAnswer)
	print("EtestAnswer: ", EtestAnswer)
	