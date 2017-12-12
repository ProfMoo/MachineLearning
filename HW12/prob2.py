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

	EinXn = 0
	#print("pls: ", X[len(X)-1].item(0,0))
	# if ((X[len(X)-1].item(0,0)) > 0):
	# 	if (point.classification == -1):
	# 		EinXn = 1
	# if ((X[len(X)-1].item(0,0)) <= 0):
	# 	#print("HERE: ", X[len(X)-1].item(0,0))
	# 	if (point.classification == 1):
	# 		EinXn = 1
	EinXn = ((X[len(X)-1].item(0,0)) - point.classification)**2

	return (GXn, EinXn)

def gradient_descent(points, W, final):
	#initialize gradient

	eta = 0.1
	beta = 0.75
	alpha = 1.05

	num_iter = 1000
	#previous_Ein = 0
	#Ein = 0
	Ein = 0
	Ein_list = list()
	num_iter_list = list()

	current_W = W[:]
	next_W = W[:]

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

	print("ein: ", Ein)

	i = 0
	while (i < num_iter):
		#printing
		if (i%50 == 0):
			print("num_iter: ", i)

		#make new weights to test

		current_W = W[:]
		next_W = W[:]

		V = list()
		j = 0
		while (j < len(G)):
			V.append(np.negative(G[j]))
			j += 1 	
		j = 0
		while (j < len(next_W)):
			next_W[j] = np.add(current_W[j], eta*V[j])
			j += 1

		current_G = G[:]
		G = [0,0,0]
		current_Ein = Ein
		Ein = 0
		j = 0
		while (j < points.getLength()):
			point = points.getPoint(j)
			GXn, EinXn = gradient_descent_point(point, next_W, final)
			k = 1
			while (k < len(G)):
				G[k] += (1/points.getLength())*GXn[k]
				k += 1
			Ein += (1/points.getLength())*EinXn
			j += 1

		if (i%10 == 0):
			num_iter_list.append(i)
			Ein_list.append(Ein)

		if (Ein < current_Ein):
			
			W = next_W[:]
			eta = alpha*eta

		if (Ein >= current_Ein):
			W = current_W[:]
			G = current_G[:]
			print("Ein: ", Ein)
			eta = beta*eta

		i += 1

	return(num_iter_list, Ein_list, W)

def makeGraph(W, starting_x1, ending_x1, starting_x2, ending_x2, increment, points):
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
	while (i < points.getLength()):
		currentPoint = points.getPoint(i)
		result_s, result_x = forward_prop(currentPoint, W, identity)
		result_x = result_x[len(result_x)-1].item(0,0)
		if (result_x > 0):
			plt.plot(currentPoint.x1, currentPoint.x2, color = "#99CCFF", linestyle = 'none', marker = 'o', label = "1")
		if (result_x <= 0):
			plt.plot(currentPoint.x1, currentPoint.x2, color = "#FF9999", linestyle = 'none', marker = 'x', label = "-1")
		i += 1

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
	starting_weights.append(np.matrix('0.01 -0.01; 0.01 -0.01; 0.01 -0.01'))
	starting_weights.append(np.matrix('0.01; 0.01; -0.01'))	
	print("tanh")
	num_iter_list, Ein_list, W = gradient_descent(points, starting_weights, identity)
	print("W: ", W)
	print("num_iter_list ", num_iter_list)
	print("Ein: ", Ein_list)

	fig = plt.figure()
	fig.suptitle('Ein', fontsize = 20)
	plt.xlabel('Iterations', fontsize = 18)
	plt.ylabel('Ein', fontsize = 18)
	plt.plot(num_iter_list, Ein_list, 'bo')
	
	x1_beg = -1.1
	x1_end = 1.1
	x2_beg = -1.1
	x2_end = 1.1
	NNpoints = makeGraph(W, x1_beg, x1_end, x2_beg, x2_end, 0.01, points)

	plt.show()
	plt.clf()
	plt.cla()

	#using W, get a decision boundary

	fig = plt.figure()
	fig.suptitle('digits', fontsize = 20)
	plt.xlabel('asymmetry', fontsize = 18)
	plt.ylabel('intensity', fontsize = 18)

	plot(NNpoints)	
	plot_points(points)

	plt.show()
	plt.clf()

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
	
	1000 iter W:
	W:  [matrix([[ 0.]]), matrix([[ 2.23506388, -2.23506388],
        [ 3.31914946, -3.31914946],
        [ 0.75665187, -0.75665187]]), matrix([[ 0.17515128],
        [-0.62428133],
        [ 0.62428133]])]
	'''
