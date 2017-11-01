import numpy
import scipy
import matplotlib

import matplotlib.pyplot as plt
from random import randint
import random

if __name__ == "__main__":
	f = open("ZipDigits.all", 'r')

	lineCount = 0
	info = []
	for line in f:
		lineCount += 1
		info.append(line)

	print("lineCount: ", lineCount)
	fileTest = open("ZipDigits.test", "w")
	fileTrain = open("ZipDigits.train", "w")

	#making test
	i = 0
	while (i < 300):
		testData = random.randint(0, lineCount)
		fileTest.write(info.pop(i))
		i += 1

	print("len(info): ", len(info))

	i = 0
	while (i < len(info)):
		fileTrain.write(info[i])
		i += 1
