import numpy
import scipy
import matplotlib

import matplotlib.pyplot as plt
from random import randint

def formula(x):
	return ((-w[1]*x)/w[2])-(w[0]/w[2])

def plot(data, h):
	i = 0
	rx = []
	ry = []
	bx = []
	by = []

	print("data: ", data)

	while (i < len(data)):
		if (h[i] == -1):
			bx.append(data[i][0])
			by.append(data[i][1])
		elif (h[i] == 1):
			rx.append(data[i][0])
			ry.append(data[i][1])
		i += 1	

	lx = numpy.linspace(0,upperBound,2000)

	gx = numpy.array(range(0, upperBound))
	gy = formula(gx)
	plt.plot(gx, gy)

	plt.plot(lx, lx)
	plt.plot(rx, ry, 'ro')
	plt.plot(bx, by, 'bx')
	plt.axis([0,upperBound,0,upperBound])
	plt.show()

x1 = []
x2 = []

data = []
red = []
blue = []
h = []
w = [-1, 0, 0]

numPoints = 20
upperBound = 100

i = 0
while (i < numPoints):
	x1pre = randint(0,upperBound)
	x2pre = randint(0,upperBound)
	if (x1pre == x2pre):
		continue
	elif (x1pre > x2pre):
		blue.append(x1pre)
		blue.append(x2pre)
		data.append(list(blue))
		blue.clear()
		h.append(-1)
	else:
		red.append(x1pre)
		red.append(x2pre)
		data.append(list(red))
		red.clear()
		h.append(1)
	i += 1

print(h)

def checkBad(k, data, h):
	insideSign = w[0]*1+w[1]*data[k][0]+w[2]*data[k][1]
	print("insideSign: ", insideSign)
	if (insideSign > 0 and h[k] > 0):
		return True
	elif (insideSign > 0 and h[k] < 0):
		return False
	elif (insideSign < 0 and h[k] < 0):
		return True
	elif (insideSign < 0 and h[k] > 0):
		return False
	#print("wut")

def updateWeights(k, data, h):
	#print("wpre: ", w)
	j = 0
	while (j < 3):
		if (j == 0):
			w[j] = w[j] + (h[k])
		else:
			w[j] = w[j] + (h[k]*data[k][j-1])
		j += 1
	#print("wafter: ", w)

numIter = 0
complete = False
while(complete == False): ##while iterations still need to be done
	i = 0
	checking = False
	while (i < len(data)): ##looping through data
		checking = checkBad(i, data, h)
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
	
print("numIter: ", numIter)
##numCheck = ((-w[1]*)/w[2])

plot(data, h)