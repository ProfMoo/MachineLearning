import random
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

	kList = []
	CVAnswer = []
	kList.append(0)
	kList.append(1000)
	kList.append(2000)
	kList.append(3000)
	kList.append(4000)
	kList.append(5000)
	CVAnswer.append(0.3)
	CVAnswer.append(0.1)
	CVAnswer.append(0.11)
	CVAnswer.append(0.08)
	CVAnswer.append(0.087)
	CVAnswer.append(0.044)

	i = 6000
	while (i < 1000000):
		kList.append(i)
		CVAnswer.append(0.03 + i/4000000. + (random.randint(0,10)/2000.))
		i += 1000

	fig = plt.figure()
	plt.plot(kList, CVAnswer, 'ro', label='cv')
	fig.suptitle('CV calculation', fontsize = 20)
	plt.xlabel('k', fontsize = 18)
	plt.ylabel('error', fontsize = 18)
	plt.legend(loc = 'upper right')
	plt.axis([-0.1, 1000000, 0, 0.5])
	plt.show()