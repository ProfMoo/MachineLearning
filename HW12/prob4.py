from sklearn import svm
import random
import numpy as np
import matplotlib.pyplot as plt

def normalizeFeatures(x1, y1, xN, yN):
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

def getIntensity(trainingDigit):
	intensity = 0
	i = 0
	while (i < 256):
		intensity += (float(trainingDigit[i]) + 1)
		i += 1
	return (intensity)

def getSymmetry(trainingDigit):
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

def get_etest(clf):
	f = open("ZipDigits.test", 'r')

	error = 0
	x1 = []
	y1 = []
	xN = []
	yN = []

	for line in f:
		line = line.split(' ')
		if (line[0] != '1.0000'):
			line = line[1:-1]
			xN.append(getSymmetry(line))
			yN.append(getIntensity(line))
		if (line[0] == '1.0000'):
			line = line[1:-1]
			x1.append(getSymmetry(line))
			y1.append(getIntensity(line))

	#normalize features
	x1, y1, xN, yN = normalizeFeatures(x1, y1, xN, yN)

	i = 0
	while (i < len(x1)):
		predict = clf.predict([[x1[i], y1[i]]])
		if (predict < 0):
			error += 1
		i +=1
	i = 0
	while (i < len(xN)):
		predict = clf.predict([[xN[i], yN[i]]])
		if (predict > 0):
			error += 1
		i +=1

	return (x1, y1, xN, yN, (error)/(len(x1)+len(xN)))

def main():
	f = open("ZipDigits.train", 'r')

	error = 0
	x1 = []
	y1 = []
	xN = []
	yN = []

	for line in f:
		line = line.split(' ')
		if (line[0] != '1.0000'):
			line = line[1:-1]
			xN.append(getSymmetry(line))
			yN.append(getIntensity(line))
		if (line[0] == '1.0000'):
			line = line[1:-1]
			x1.append(getSymmetry(line))
			y1.append(getIntensity(line))

	#normalize features
	x1, y1, xN, yN = normalizeFeatures(x1, y1, xN, yN)

	x_mat = []
	y_mat = []

	i = 0
	while (i < len(x1)):
		x1[i] = np.float32(x1[i])
		y1[i] = np.float32(y1[i])
		x_mat.append([x1[i], y1[i]])
		y_mat.append(1)
		i += 1

	i = 0
	while (i < len(xN)):
		xN[i] = np.float32(xN[i])
		yN[i] = np.float32(yN[i])
		x_mat.append([xN[i], yN[i]])
		y_mat.append(-1)
		i += 1

	c = 1.5
	clf = svm.SVC(c)
	clf.fit(x_mat, y_mat)

	# predicting = np.array([[(2), (-1)]])
	# print(clf.predict(predicting).item(0))
	graphP_x = []
	graphP_y = []
	graphR_x = []
	graphR_y = []

	#decision boundaries
	x_it = -1
	y_it = -1
	while x_it <= 1:
		predict = clf.predict([[x_it, y_it]])
		# print predict
		if predict.item(0) > 0:
			graphP_x.append(x_it)
			graphP_y.append(y_it)
		else:
			graphR_x.append(x_it)
			graphR_y.append(y_it)
		y_it += .02
		if y_it >= 1:
			y_it = -1
			x_it += .02

	t1, t2, t3, t4, etest = get_etest(clf)
	print("etest: ", etest)
	print("c: ", c)

	plt.xlabel('asymmetry')
	plt.ylabel('intensity')

	plt.plot(graphP_x, graphP_y, color = "#99CCFF", linestyle = 'none', marker = 'o', label = "1")
	plt.plot(graphR_x, graphR_y, color = "#FF9999", linestyle = 'none', marker = 'x', label = "-1")

	#train
	plt.plot(x1, y1, 'bo', label='1')
	plt.plot(xN, yN, 'ro', label='not 1')
	#test
	# plt.plot(t1, t2, 'bo', label='1')
	# plt.plot(t3, t4, 'ro', label='not 1')

	plt.show()

if __name__ == "__main__":
	main()

