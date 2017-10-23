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
	print("mins: ", mins)
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

	print("minsList: ", minsList)
	print("xs: ", xs)
	print("final x: ", xt)
	print("final y: ", yt)

	g, = plt.plot(xs,minsList,'ro',label='f(x,y)')
	plt.xlabel('Steps')
	plt.ylabel('f(x,y)')
	plt.legend(handles=[g])
	plt.show()

if __name__ == "__main__":
	eta = 0.1
	startx = 0.1
	starty = 0.1
	numIter = 50

	run(eta, startx, starty, numIter)
