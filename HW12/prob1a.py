import numpy as np
import scipy
import matplotlib
import math

import matplotlib.pyplot as plt
from random import randint
import random

def tanh(x):
	return np.tanh(x)

def tanhprime(x):
	return 1-((x)**2)

def forward_prop(x1, y, W):
	S = list()
	X = list()
	X.append(x1)
	vec_op = np.vectorize(tanh)

	i = 0
	while (i < len(W)):
		#print("i: ", i)
		S.append(np.dot(np.transpose(W[i]), X[i]))
		#print("S[i]: ", S[i])
		vec_matrix = vec_op(S[i])
		#print("vec_matrix: ", vec_matrix)
		X.append(np.transpose(np.matrix(np.append([1], vec_matrix))))
		i += 1

	S.insert(0, np.matrix(0))
	W.insert(0, np.matrix(0))
	X[len(X)-1] = X[len(X)-1][1:]
	return(S, X)

def backward_prop(S, X, W, y):
	D = list()
	vec_op = np.vectorize(tanhprime)
	D.append(np.matrix( 2 * (X[len(X)-1].item(0,0) - y) * (vec_op(X[len(X)-1]) ) ) )

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

def gradient_descent_point(x1, y, W, G):
	S, X = forward_prop(x1, y, W)
	print("S: ", S)
	print("X: ", X)
	print("W: ", W)
	D = backward_prop(S, X, W, y)
	print("D: ", D)

	#compute gradient
	GXn = [0,0,0]

	i = 1
	while (i < len(X)):
		GXn[i] = X[i-1]*np.transpose(D[i])
		i += 1

	print("GXn: ", GXn)

def gradient_descent(x1, y, W):
	#initialize gradient
	G = [0,0]

	gradient_descent_point(x1, y, W, G)

	#while loop, calling point for GD

def main():
	x1 = np.matrix('1;1;1')
	y = 1
	starting_weights = list()
	starting_weights.append(np.matrix('0.25 0.25; 0.25 0.25; 0.25 0.25'))
	starting_weights.append(np.matrix('0.25; 0.25; 0.25'))
	#print(x1)
	#print(y)
	#print(starting_weights)
	
	gradient_descent(x1, y, starting_weights)

if __name__ == "__main__":
	main()
