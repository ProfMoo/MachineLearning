import numpy
import scipy
import matplotlib

import matplotlib.pyplot as plt
from random import randint
import random

def deriv(eta, x):
	x -= eta*(2*x+2*numpy.pi*numpy.cos(2*numpy.pi*x))
	return x

def getMins(x, y):
	mins = x ** 2 + y ** 2 + 2 * numpy.sin(2 * numpy.pi * x) + 2 * numpy.sin(2 * numpy.pi * y)
	return mins

def run(eta, startingx, startingy, numIter):
	minsList = []
	xs = []
	xt = startingx
	yt = startingy

	i = 0
	while(i < numIter):
		minsList.append(getMins(xt, yt))
		xt = deriv(eta, xt)
		yt = deriv(eta, yt)
		xs.append(i)
		i += 1

	print("minsList[-1]: ", minsList[-1])

if __name__ == "__main__":
	numIter = 50

	print("running eta=0.1, location=0.1")
	run(0.1, 0.1, 0.1, numIter)
	print("running eta=0.01, location=0.1")
	run(0.01, 0.1, 0.1, numIter)
	print("running eta=0.1, location=1")
	run(0.1, 1, 1, numIter)
	print("running eta=0.01, location=1")
	run(0.01, 1, 1, numIter)
	print("running eta=0.1, location=-0.5")
	run(0.1, -0.5, -0.5, numIter)
	print("running eta=0.01, location=-0.5")
	run(0.01, -0.5, -0.5, numIter)
	print("running eta=0.1, location=-1")
	run(0.1, -1, -1, numIter)
	print("running eta=0.01, location=-1")
	run(0.01, -1, -1, numIter)
