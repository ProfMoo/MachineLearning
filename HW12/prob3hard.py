import numpy as np
import matplotlib.pyplot as plt
import cvxopt

from Point import *
from PointHolder import *

def qp():
	Q = np.matrix('0 0 0; 0 1 0; 0 0 1')
	p = np.matrix('0; 0; 0')
	A = np.matrix('1 1 0; -1 1 0')
	c = np.matrix('1; 1')

	Q = Q.astype(np.double)
	p = p.astype(np.double)
	A = A.astype(np.double)
	c = c.astype(np.double)

	print("Q: ", Q)
	print("p: ", p)
	print("A: ", A)
	print("c: ", c)

	Q = cvxopt.matrix(Q)
	p = cvxopt.matrix(p)
	A = cvxopt.matrix(-A)
	c = cvxopt.matrix(-c)

	u = cvxopt.solvers.qp(Q,p,A,c)

	print(u['x'])
	return(u['x'][0], u['x'][1:])

def check_func(b, W, point):
	check_num = np.dot( np.transpose(np.matrix([[W[0]], [W[1]]])), np.matrix([[point.x1], [point.x2]]) ) + b
	if (check_num > 0.001):
		return 1
	if (check_num < -0.001):
		return -1
	return 0

def check_func_Z(b, W, point):
	check_num = np.dot( np.transpose(np.matrix([[W[0]], [W[1]]])), np.matrix([[point.x1**3-point.x2], [point.x1*point.x2]]) ) + b
	if (check_num > 0.02):
		return 1
	if (check_num < -0.02):
		return -1
	return 0

def make_chart(b, W, x1_beg, x1_end, x2_beg, x2_end, increment, type_func):
	SVM_points = PointHolder()

	#looping through all locations in NN graph
	i = x1_beg
	while (i < x1_end):
		j = x2_beg
		while (j < x2_end):
			new_point = Point(i, j, 0)

			if (type_func == "X"):
				answer = check_func(b, W, new_point)
			if (type_func == "Z"):
				answer = check_func_Z(b, W, new_point)

			if (answer > 0):
				SVM_points.addPoint(Point(i, j, 1))
			if (answer < 0):
				SVM_points.addPoint(Point(i, j, -1))
			if (answer == 0):
				SVM_points.addPoint(Point(i, j, 0))
			j += increment
		i += increment

	return (SVM_points)

def plot(SVM_points):
	i = 0
	while (i < SVM_points.getLength()):
		currentPoint = SVM_points.getPoint(i)
		if (currentPoint.classification == 1):
			plt.plot(currentPoint.x1, currentPoint.x2, color = "#99CCFF", linestyle = 'none', marker = 'x', label = "1")
		elif (currentPoint.classification == -1):
			plt.plot(currentPoint.x1, currentPoint.x2, color = "#FF9999", linestyle = 'none', marker = 'x', label = "-1")
		elif (currentPoint.classification == 0):
			plt.plot(currentPoint.x1, currentPoint.x2, color = "#000000", linestyle = 'none', marker = 'x', label = "0")
		i += 1

def plot_points(points):
	i = 0
	while (i < points.getLength()):
		currentPoint = points.getPoint(i)
		if (currentPoint.classification == 1):
			plt.plot(currentPoint.x1, currentPoint.x2, 'bo')
		elif (currentPoint.classification == -1):
			plt.plot(currentPoint.x1, currentPoint.x2, 'rx')
		i += 1

def main():
	b, W = qp()

	x1_beg = -1.0
	x1_end = 1.0
	x2_beg = -1.0
	x2_end = 1.0
	type_func = "Z"
	SVM_points = make_chart(b, W, x1_beg, x1_end, x2_beg, x2_end, 0.025, type_func)

	plot(SVM_points)

	points = PointHolder()
	points.addPoint(Point(1, 0, 1))
	points.addPoint(Point(-1, 0, -1))
	plot_points(points)
	plt.show()

if __name__ == "__main__":
	main()