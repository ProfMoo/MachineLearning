class Point(object):
	def __init__(self, Ix1, Ix2, Iclassification):
		self.one = 1
		self.x1 = Ix1
		self.x2 = Ix2
		self.classification = Iclassification

	def __str__(self):
		return "Values: " + str(self.x1) + " " + str(self.x2)