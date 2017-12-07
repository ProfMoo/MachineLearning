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
	i = 0
	S = list()
	X = list()
	X.append(x1)
	vec_op = np.vectorize(tanh)
	while (i < len(W)):
		S.append(np.dot(np.transpose(W[i]), X[i]))
		vec_matrix = vec_op(S[i])
		X.append(np.transpose(np.matrix(np.append([1], vec_matrix))))
		i += 1

	return(S, X)

def backward_prop(S, X, W, y):
	D = list()
	D.append( 2 * (X[0].item(0,0) - y) * tanhprime(S[len(S)-1]))
	i = 0
	while (i < len(W)):
		#need two loops in here

		#get theta prime here
		#get sensitivty here and append 
		i += 1

def gradient_descent(x1, y, W):
	S, X = forward_prop(x1, y, W)
	print("S: ", S)
	print("X: ", X)
	print("W: ", W)
	backward_prop(S, X, W, y)

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
