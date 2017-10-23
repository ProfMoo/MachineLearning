import numpy
import scipy
import matplotlib

import matplotlib.pyplot as plt
from random import randint
import random

w = [-1,1,1,1,1,1,1,1,1,1]

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

def getData(x1, y1, x5, y5):
	databack = []
	singledata = []
	i = 0
	while (i < len(x1)):
		singledata = []
		singledata.append(1)
		singledata.append(x1[i])
		singledata.append(y1[i])
		singledata.append(x1[i]**2)
		singledata.append(x1[i]*y1[i])
		singledata.append(y1[i]**2)
		singledata.append(x1[i]**3)
		singledata.append((x1[i]**2)*y1[i])
		singledata.append(x1[i]*(y1[i]**2))
		singledata.append(y1[i]**3)
		databack.append(singledata)
		i += 1

	i = 0
	while (i < len(x5)):
		singledata = []
		singledata.append(1)
		singledata.append(x5[i])
		singledata.append(y5[i])
		singledata.append(x5[i]**2)
		singledata.append(x5[i]*y5[i])
		singledata.append(y5[i]**2)
		singledata.append(x5[i]**3)
		singledata.append((x5[i]**2)*y5[i])
		singledata.append(x5[i]*(y5[i]**2))
		singledata.append(y5[i]**3)
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
	while (j < len(w)):
		w[j] = w[j] + (h[k]*data[k][j])
		j += 1

def runSimulation(h, data):
	numIter = 0
	complete = False
	wbest = w[:]
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

	pocketTop = 10000
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
	data = getData(x1, y1, x5, y5)
	#print("data: ", data)
	ys = getYs(x1, x5)

	if (whichFile == "train"):
		#get wlin
		x_matrix = numpy.matrix(data)
		y_matrix = numpy.matrix(ys)
		print("x_matrix: ", x_matrix)
		print("y_matrix: ", y_matrix)
		x_matrix_trans = numpy.transpose(x_matrix)
		wlin = (numpy.linalg.inv(x_matrix_trans * x_matrix) * x_matrix_trans) * numpy.transpose(y_matrix)
		print("wlin: ", wlin)

		i = 0
		while (i < len(w)):
			w[i] = wlin[i].item(0,0)
			i += 1

		wbest = runSimulation(ys, data)

	#if training, put info here
	if (whichFile == "test"):
		wbest = [3900.9537672701094, 132444.29258090307, 945.3587626474581, 3956844.6179376612, 29690.85822605399, 224.39950620190498, -44418.04716501154, 391730.36874999397, 4700.622216805807, 35.35717958521439]

	x11 = numpy.linspace(0, 275, 250)
	x22 = numpy.linspace(0, 1.25, 250)
	x11, x22 = numpy.meshgrid(x11, x22)	

	#now, plot
	fig = plt.figure()
	plt.plot(x5, y5, 'rx')
	plt.plot(x1, y1, 'bo')
	plt.contour(x11, x22, wbest[0] + wbest[1]*x11 + wbest[2]*x22 + wbest[3]*x11**2 + wbest[4]*x11*x22 + wbest[5]*x22**2 + wbest[6]*x11**3 + wbest[7]*x11**2*x22 + wbest[8]*x11*x22**2 + wbest[9]*x22**3, [0])

	#calculate Ein
	print("wbest: ", wbest)
	Ein = EinCalc(data, ys, wbest)
	print("Ein: ", Ein)

	fig.suptitle('digits', fontsize = 20)
	plt.xlabel('asymmetry', fontsize = 18)
	plt.ylabel('intensity', fontsize = 16)
	plt.show()