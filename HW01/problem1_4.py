import numpy
import scipy
import matplotlib

import matplotlib.pyplot as plt
from random import randint

x1 = []
x2 = []

rx = []
ry = []
bx = []
by = []
h = []

numPoints = 20
upperBound = 100

i = 0
while (i < numPoints):
	x1pre = randint(0,upperBound)
	x2pre = randint(0,upperBound)
	if (x1pre == x2pre):
		continue
	elif (x1pre > x2pre):
		bx.append(x1pre)
		by.append(x2pre)
		h.append(-1)
	else:
		rx.append(x1pre)
		ry.append(x2pre)
		h.append(1)
	i += 1

print(h)

lx = numpy.linspace(0,upperBound,2000)

plt.plot(lx, lx)
plt.plot(rx, ry, 'ro')
plt.plot(bx, by, 'bx')
plt.axis([0,upperBound,0,upperBound])
plt.show()