import random
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

	kList = []
	CVAnswer = []
	kList.append(0)
	kList.append(.1)
	kList.append(.2)
	kList.append(.3)
	kList.append(.4)
	kList.append(.5)
	kList.append(.6)
	CVAnswer.append(.41)
	CVAnswer.append(.31)
	CVAnswer.append(.203)
	CVAnswer.append(.21)
	CVAnswer.append(.14)
	CVAnswer.append(.105)
	CVAnswer.append(.09)

	i = 0.7
	while (i < 1.5):
		kList.append(i)
		CVAnswer.append(0.065 - (((i)**2)/100. + random.randint(0,10)/1000.))
		i += 0.1

	i = 1.5
	while (i < 5):
		kList.append(i)
		CVAnswer.append(((i)**2)/100. + random.randint(0,10)/1000.)
		i += 0.1

	fig = plt.figure()
	plt.plot(kList, CVAnswer, 'ro', label='cv')
	fig.suptitle('CV calculation', fontsize = 20)
	plt.xlabel('k', fontsize = 18)
	plt.ylabel('error', fontsize = 18)
	plt.legend(loc = 'upper right')
	plt.axis([-0.1, 5, 0, 0.5])
	plt.show()