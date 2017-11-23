class NNOB(object):

	def __init__(self, Value, Classification):
		self.value = Value
		self.classification = Classification

	def __str__(self):
		return "Value " + str(self.value) + " Classification: " + str(self.classification)

	def __lt__(self, other):
		return (self.value < other.value)

	def __gt__(self, other):
		return (self.value > other.value)