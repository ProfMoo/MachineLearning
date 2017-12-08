import numpy as np
import scipy
import matplotlib
import math

import matplotlib.pyplot as plt
from random import randint
import random

def tanh(x):
	return x
	#return np.tanh(x)

def tanhprime(x):
	return 1

def forward_prop(x1, y, W):
	S = list()
	X = list()
	X.append(x1)
	vec_op = np.vectorize(tanh)

	i = 0
	while (i < len(W)):
		S.append(np.dot(np.transpose(W[i]), X[i]))
		vec_matrix = vec_op(S[i])
		X.append(np.transpose(np.matrix(np.append([1], vec_matrix))))
		i += 1

	return(S, X)

def backward_prop(S, X, W, y):
	D = list()
	D.append(np.matrix( 2 * (X[0].item(0,0) - y) * tanhprime(S[len(S)-1]) ) )
	print("D: ", D)
	vec_op = np.vectorize(tanhprime)

	i = len(W)-1
	print("i: ", i)
	while (i > 0):
		#make theta
		theta = (vec_op(X[i]))[0:2]
		print(theta)
		
		#get sensitivity
		print("W[i]: ", W[i])
		print("D[i-1]: ", D[i-1])
		sensitivity = np.multiply(theta, (np.dot(W[i], D[i-1]))[0:2])
		D.insert(0, sensitivity)
		i -= 1

	return D

def gradient_descent(x1, y, W):
	S, X = forward_prop(x1, y, W)
	print("S: ", S)
	print("X: ", X)
	print("W: ", W)
	D = backward_prop(S, X, W, y)

	print("D: ", D)

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
