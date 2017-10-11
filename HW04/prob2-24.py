import matplotlib.pyplot as plt
import numpy as np
import random

numDataPoints = 100

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

def graphvar(varx):
	a = []
	b = []
	x = -1.
	while (x < 1.):
		y = ((varx[0])*(x**2))+(varx[1])*x+(varx[2])
		a.append(x)
		b.append(y)
		x += 0.01

	ax.plot(a,b,'y')

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

def getGdx(Gbar, MsBs):
	i = 0
	GdxMs = []
	GdxBs = []
	while (i < numDataPoints):
		GdxMs.append(MsBs[0][i] - Gbar[0])
		GdxBs.append(MsBs[1][i] - Gbar[1])
		i += 1

	return(GdxMs, GdxBs)

def getvarx(Gdx):
	i = 0
	quads = []
	while (i < numDataPoints):
		singleQuad = []
		singleQuad.append(Gdx[0][i] * Gdx[0][i])
		singleQuad.append(Gdx[0][i] * Gdx[1][i] * 2)
		singleQuad.append(Gdx[1][i] * Gdx[1][i])
		quads.append(singleQuad)
		i += 1

	#now add them all up and divide by num points
	a = 0
	ab = 0
	b = 0
	i = 0
	while (i < numDataPoints):
		a += quads[i][0]
		ab += quads[i][1]
		b += quads[i][2]
		i += 1

	a /= numDataPoints
	ab /= numDataPoints
	b /= numDataPoints

	return(a,ab,b)


if __name__ == "__main__":
	neg1to1 = np.arange(-1., 1., 0.05)
	fig,ax = plt.subplots()

	#making a pseudorandom set of data points
	dataSet = generatePoints()
	print("dataSet: ", dataSet)

	#get all slopes and y-intercepts
	MsBs = getMBs(dataSet)
	print("MsBs: ", MsBs)

	#plot all the dataset gs 
	plotLines(MsBs, neg1to1)

	print("====================================")

	#getting gbar
	Gbar = getGbar(MsBs)
	plotLine(Gbar[0], Gbar[1], neg1to1, 1)
	print("Gbar slope: ", Gbar[0])
	print("Gbar y-intercept: ", Gbar[1])

	print("====================================")

	#getting Gdx values
	Gdx = getGdx(Gbar, MsBs)
	print("GdxMs: ", Gdx[0])
	print("GdxBs: ", Gdx[1])

	print("====================================")

	#getting varx
	varx = getvarx(Gdx)
	print("varxa: ", varx[0])
	print("varxab: ", varx[1])
	print("varxb: ", varx[2])

	#graphing variance
	graphvar(varx)

	#plotting f(x)
	ax.plot(neg1to1, neg1to1*neg1to1, 'b')


	print("====================================")

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
