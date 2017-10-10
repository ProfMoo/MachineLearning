import matplotlib.pyplot as plt
import numpy as np

def plotLine(m, b, x):
	ax.plot(x, x*m+b, 'r')

if __name__ == "__main__":
	neg1to1 = np.arange(-1., 1., 0.05)

	fig,ax = plt.subplots()
	plotLine(2, 1, neg1to1)
	ax.set_aspect('equal')
	ax.grid(True, which='both')

	# set the x-spine (see below for more info on `set_position`)
	ax.spines['left'].set_position('zero')

	# turn off the right spine/ticks
	ax.spines['right'].set_color('none')
	ax.yaxis.tick_left()

	# set the y-spine
	ax.spines['bottom'].set_position('zero')

	# turn off the top spine/ticks
	ax.spines['top'].set_color('none')
	ax.xaxis.tick_bottom()

	plt.show()
