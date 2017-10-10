import matplotlib.pyplot as plt
import numpy as np
import random

numDataPoints = 10

def plotLine(m, b, x, color):
	if (color == 0):
		ax.plot(x, x*m+b, 'r')
	if (color == 1):
		ax.plot(x, x*m+b, 'g')

def plotLines(MsBs, x):
	i = 0
	while (i < numDataPoints):
		plotLine(MsBs[0][i], MsBs[1][i], x, 0)
		i += 1

def singleNum():
	return random.uniform(-1.,1.)

def generatePoints():
	i = 0
	dataSet = []
	while (i < numDataPoints):
		singleSet = []
		singleNum1 = singleNum()
		singleNum2 = singleNum()
		singleSet1 = (singleNum1, singleNum1*singleNum1)
		singleSet2 = (singleNum2, singleNum2*singleNum2)
		singleSet.append(singleSet1)
		singleSet.append(singleSet2)
		dataSet.append(singleSet)
		i += 1

	return dataSet

def getMBs(dataSet):
	Ms = []
	Bs = []
	i = 0
	while (i < numDataPoints):
		m = (dataSet[i][1][1] - dataSet[i][0][1])/(dataSet[i][1][0] - dataSet[i][0][0])
		b = (dataSet[i][0][1]) - (m*(dataSet[i][0][0]))
		Ms.append(m)
		Bs.append(b)
		i += 1

	return (Ms, Bs)

def getGbar(MsBs):
	i = 0
	addMs = 0
	addBs = 0
	while (i < numDataPoints):
		addMs += MsBs[0][i]
		addBs += MsBs[1][i]
		i += 1

	return(addMs/numDataPoints, addBs/numDataPoints)


if __name__ == "__main__":
	neg1to1 = np.arange(-1., 1., 0.05)

	fig,ax = plt.subplots()
	ax.plot(neg1to1, neg1to1*neg1to1, 'b')

	dataSet = generatePoints()
	print("dataSet: ", dataSet)

	MsBs = getMBs(dataSet)
	print("MsBs: ", MsBs)

	plotLines(MsBs, neg1to1)

	Gbar = getGbar(MsBs)
	plotLine(Gbar[0], Gbar[1], neg1to1, 1)

	##plotLine(2, 1, neg1to1)
	ax.set_aspect('equal')
	ax.grid(True, which='both')

	# set the x-spine (see below for more info on `set_position`)
	ax.spines['left'].set_position('zero')

	# turn off the right spine/ticks
	ax.spines['right'].set_color('none')
	ax.yaxis.tick_left()

	# set the y-spine
	ax.spines['bottom'].set_position('zero')

	# turn off the top spine/ticks
	ax.spines['top'].set_color('none')
	ax.xaxis.tick_bottom()

	plt.show()
