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

	def getNewCenter(self, other):
		i = 0
		maxDistance = 0
		print("other.getLength(): ", other.getLength())

		i = 0
		maxDistance = 0
		while (i < self.getLength()):
			j = 0
			loopDistance = 65536
			while (j < other.getLength()):
				distance = getDistance(self.getPoint(i), other.getPoint(j))
				if (distance < loopDistance):
					loopDistance = distance
				j += 1
			if (loopDistance > maxDistance):
				maxPoint = self.getPoint(i)
			i += 1
			
		return maxPoint

def getDistance(point1, point2):
	return math.sqrt( ((point1.x1 - point2.x1)**2) + ((point1.x2 - point2.x2)**2) )