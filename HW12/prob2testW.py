import numpy as np
import scipy
import matplotlib
import math

import matplotlib.pyplot as plt
from random import randint
import random

from Point import *
from PointHolder import *

def identity(x):
	return x

def one(x):
	return 1

def tanh(x):
	return np.tanh(x)

def tanhprime(x):
	return 1-((x)**2)

def get_intensity(trainingDigit):
	intensity = 0
	i = 0
	while (i < 256):
		intensity += (float(trainingDigit[i]) + 1)
		i += 1
	return (intensity)

def get_symmetry(trainingDigit):
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

def normalize_features(x1, y1, xN, yN):
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

def organize_data(x1, y1, xN, yN):
	points = PointHolder()
	i = 0
	while (i < len(x1)):
		points.addPoint(Point(x1[i], y1[i], 1))
		i += 1	

	i = 0
	while (i < len(xN)):
		points.addPoint(Point(xN[i], yN[i], -1))
		i += 1

	return points

def forward_prop(point, W, final):
	S = list()
	X = list()
	X.append(np.matrix([[1], [point.x1], [point.x2]]))
	vec_op = np.vectorize(tanh)
	vec_op_final = np.vectorize(final)

	i = 0
	while (i < len(W)-1):
		S.append(np.dot(np.transpose(W[i+1]), X[i]))
		#print("S[i]: ", S[i])
		if (i == len(W)-2):
			vec_matrix = vec_op_final(S[i])
		else: 
			vec_matrix = vec_op(S[i])
		#print("vec_matrix: ", vec_matrix)
		X.append(np.transpose(np.matrix(np.append([1], vec_matrix))))
		i += 1

	S.insert(0, np.matrix(0))
	X[len(X)-1] = X[len(X)-1][1:]
	return(S, X)

def makeGraph(W, starting_x1, ending_x1, starting_x2, ending_x2, increment):
	NNpoints = PointHolder()

	#looping through all locations in NN graph
	i = starting_x1
	while (i < ending_x1):
		# if (i%1 < 0.01):
		# 	print("i: ", i)
		j = starting_x2
		while (j < ending_x2):
			new_point = Point(i, j, 0)
			new_s, new_x = forward_prop(new_point, W, tanh)
			result = new_x[len(new_x)-1].item(0, 0)
			#print("result" , result)
			#print("newpoint: ", new_point)
			if (result > 0):
				NNpoints.addPoint(Point(i, j, 1))
			if (result <= 0):
				NNpoints.addPoint(Point(i, j, -1))
			j += increment
		i += increment

	return (NNpoints)

def get_data():
	f = open("ZipDigits.test", 'r')

	x1 = []
	y1 = []
	xN = []
	yN = []

	for line in f:
		line = line.split(' ')
		if (line[0] != '1.0000'):
			line = line[1:-1]
			xN.append(get_symmetry(line))
			yN.append(get_intensity(line))
		if (line[0] == '1.0000'):
			line = line[1:-1]
			x1.append(get_symmetry(line))
			y1.append(get_intensity(line))

	#normalize features
	x1, y1, xN, yN = normalize_features(x1, y1, xN, yN)

	points = organize_data(x1, y1, xN, yN)
	return points

def plot_points(points):
	i = 0
	while (i < points.getLength()):
		currentPoint = points.getPoint(i)
		if (currentPoint.classification == 1):
			plt.plot(currentPoint.x1, currentPoint.x2, 'bo')
		elif (currentPoint.classification == -1):
			plt.plot(currentPoint.x1, currentPoint.x2, 'rx')
		i += 1

def plot_test(points, W):
	i = 0
	error = 0
	while (i < points.getLength()):
		currentPoint = points.getPoint(i)
		result_s, result_x = forward_prop(currentPoint, W, identity)
		result_x = result_x[len(result_x)-1].item(0,0)
		if (result_x > 0 and currentPoint.classification == -1):
			error += 1
		if (result_x <= 0 and currentPoint.classification == 1):
			error += 1
		#if (result_x > 0):
			#plt.plot(currentPoint.x1, currentPoint.x2, color = "#99CCFF", linestyle = 'none', marker = 'o', label = "1")
		#if (result_x <= 0):
			#plt.plot(currentPoint.x1, currentPoint.x2, color = "#FF9999", linestyle = 'none', marker = 'x', label = "-1")
		i += 1

	print("Error: ", error/points.getLength())

def plot(NNpoints):
	i = 0
	while (i < NNpoints.getLength()):
		currentPoint = NNpoints.getPoint(i)
		if (currentPoint.classification == 1):
			plt.plot(currentPoint.x1, currentPoint.x2, color = "#99CCFF", linestyle = 'none', marker = 'x', label = "1")
		elif (currentPoint.classification == -1):
			plt.plot(currentPoint.x1, currentPoint.x2, color = "#FF9999", linestyle = 'none', marker = 'x', label = "-1")
		i += 1

def main():
	points = get_data()

	W = [np.matrix([[ 0.]]), np.matrix([[ 3.89664268, -3.89664268],
        [ 6.59767342, -6.59767342],
        [ 1.45769226, -1.45769226]]), np.matrix([[ 0.0056534 ],
        [-0.51060269],
        [ 0.51060269]])]
	
	x1_beg = -1.1
	x1_end = 1.1
	x2_beg = -1.1
	x2_end = 1.1
	NNpoints = makeGraph(W, x1_beg, x1_end, x2_beg, x2_end, 0.01)


	#using W, get a decision boundary

	fig = plt.figure()
	fig.suptitle('digits', fontsize = 20)
	plt.xlabel('asymmetry', fontsize = 18)
	plt.ylabel('intensity', fontsize = 18)

	plot(NNpoints)	
	plot_points(points)
	plot_test(points, W)
	plt.show()

if __name__ == "__main__":
	main()

	'''
	10000 iter W:
	W:  [matrix([[ 0.]]), matrix([[ 3.89664268, -3.89664268],
        [ 6.59767342, -6.59767342],
        [ 1.45769226, -1.45769226]]), matrix([[ 0.0056534 ],
        [-0.51060269],
        [ 0.51060269]])]
	'''
