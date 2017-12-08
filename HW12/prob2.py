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

def backward_prop(S, X, W, point, final):
	D = list()
	vec_op = np.vectorize(tanhprime)
	if (final == identity):
		vec_op_final = np.vectorize(one)
	else:
		vec_op_final = np.vectorize(tanhprime)
	D.append(np.matrix( 2 * (X[len(X)-1].item(0,0) - point.classification) * (vec_op_final(X[len(X)-1]) ) ) )

	i = len(S)-2
	while (i > 0):
		#make theta
		theta = np.matrix((vec_op(X[i]))[1:])
		
		#get sensitivity
		sensitivity = np.multiply(theta, (np.dot(W[i+1], D[i-1]))[1:])
		D.insert(0, sensitivity)
		i -= 1

	D.insert(0, np.matrix(0))
	return D

def gradient_descent_point(point, W, final):
	S, X = forward_prop(point, W, final)
	# print("S: ", S)
	# print("X: ", X)
	# print("W: ", W)
	D = backward_prop(S, X, W, point, final)
	# print("D: ", D)

	#compute gradient
	GXn = [0,0,0]

	i = 1
	while (i < len(X)):
		GXn[i] = X[i-1]*np.transpose(D[i])
		i += 1

	EinXn = (X[len(X)-1].item(0,0) - point.classification)**2

	return (GXn, EinXn)

def gradient_descent(points, W, final):
	#initialize gradient

	eta = 0.01
	beta = -0.08
	alpha = 1.03

	num_iter = 400
	previous_Ein = 0
	Ein = 0
	Ein_list = list()
	num_iter_list = list()

	#do first iter to get G
	G = [0,0,0]
	j = 0
	while (j < points.getLength()):
		point = points.getPoint(j)
		GXn, EinXn = gradient_descent_point(point, W, final)
		k = 1
		while (k < len(G)):
			G[k] += (1/points.getLength())*GXn[k]
			k += 1
		Ein += (1/points.getLength())*EinXn
		j += 1

	i = 0
	while (i < num_iter):
		#printing
		if (i%50 == 0):
			print("num_iter: ", i)

		#make new weights to test
		previous_W = W[:]

		V = list()
		j = 0
		while (j < len(G)):
			V.append(np.negative(G[j]))
			j += 1 	
		j = 0
		while (j < len(W)):
			W[j] = np.add(W[j], eta*V[j])
			j += 1
		#updates on bad Gs no matter what!!!

		previous_G = G[:]
		G = [0,0,0]
		previous_Ein = Ein
		Ein = 0
		j = 0
		while (j < points.getLength()):
			point = points.getPoint(j)
			GXn, EinXn = gradient_descent_point(point, W, final)
			k = 1
			while (k < len(G)):
				G[k] += (1/points.getLength())*GXn[k]
				k += 1
			Ein += (1/points.getLength())*EinXn
			j += 1

		if (i%10 == 0):
			num_iter_list.append(i)
			Ein_list.append(Ein)

		if (Ein < previous_Ein):
			eta = alpha*eta

		if (Ein >= previous_Ein):
			print("BAD")
			W = previous_W[:]
			G = previous_G[:]
			eta = beta*eta

		i += 1

	print("num_iter_list ", num_iter_list)
	print("Ein: ", Ein_list)
	return(num_iter_list, Ein_list)
	#get final GD


def get_data():
	f = open("ZipDigits.train", 'r')

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
	fig = plt.figure()
	fig.suptitle('digits', fontsize = 20)
	plt.xlabel('asymmetry', fontsize = 18)
	plt.ylabel('intensity', fontsize = 18)

	i = 0
	while (i < points.getLength()):
		currentPoint = points.getPoint(i)
		if (currentPoint.classification == 1):
			plt.plot(currentPoint.x1, currentPoint.x2, 'bo')
		elif (currentPoint.classification == -1):
			plt.plot(currentPoint.x1, currentPoint.x2, 'rx')
		i += 1

def main():
	points = get_data()
	#plot_points(points)

	# x1 = np.matrix('1;1;1')
	# y = 1
	# starting_weights = list()
	# starting_weights.append(np.matrix('0.25 0.25; 0.25 0.25; 0.25 0.25'))
	# starting_weights.append(np.matrix('0.25; 0.25; 0.25'))	
	# print("identity")
	# gradient_descent(points, starting_weights, identity)

	x1 = np.matrix('1;1;1')
	y = 1
	starting_weights = list()
	starting_weights.append(np.matrix(0))
	starting_weights.append(np.matrix('0.25 0.25; 0.25 0.25; 0.25 0.25'))
	starting_weights.append(np.matrix('0.25; 0.25; 0.25'))	
	print("tanh")
	num_iter_list, Ein_list = gradient_descent(points, starting_weights, tanh)

	plt.plot(num_iter_list, Ein_list, 'bo')

	plt.show()

if __name__ == "__main__":
	main()
