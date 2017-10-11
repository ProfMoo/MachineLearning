import matplotlib.pyplot as plt
import numpy as np
import random

if __name__ == "__main__":
	zeroto1 = np.arange(0., 1., 0.01)
	fig,ax = plt.subplots()
	ax.plot(zeroto1, 1-zeroto1, 'b')

	ax.fill_between(zeroto1,1.,1.)

	plt.show()
