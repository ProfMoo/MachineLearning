import numpy
import scipy
import matplotlib

import matplotlib.pyplot as plt

def showImage(digit):

	i = 0
	while (i < len(digit)):
		digit[i] = float(digit[i])
		i += 1

	img = numpy.reshape(digit, (-1, 16))
	plt.imshow(img, cmap='gray_r', interpolation='nearest')
	plt.show()

if __name__ == "__main__":
	f = open("ZipDigits.train", 'r')

	for line in f:
		line = line.split(' ')
		if (line[0] == '1.0000'):
			line = line[1:-1]
			showImage(line)
			break