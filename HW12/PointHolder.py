import math

class PointHolder(object):
	def __init__(self):
		self.data = []

	def __str__(self):
		i = 0
		returnStr = ""
		while (i < len(self.data)):
			returnStr += "Point " + str(i) + " " + str((self.data)[i].x1) + " " + str((self.data)[i].x2) + "\n"
			i += 1
		return returnStr

	def getLength(self):
		return len(self.data)

	def addPoint(self, point):
		(self.data).append(point)

	def getPoint(self, i):
		return (self.data)[i]