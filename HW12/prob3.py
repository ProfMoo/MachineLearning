import numpy as np
import matplotlib.pyplot as plt

x_x1 = []
x_x2 = []
z_x1 = []
z_x2 = []

i = -1
while (i < 1.1):
	j = -1
	while (j < 1.1):
		#print("i: ", i)
		if (-0.01 < i < 0.01):
			x_x1.append(i)
			x_x2.append(j)
		if (-0.01 < np.power(i,3) - j < 0.01):
			z_x1.append(i)
			z_x2.append(j)
		j += 0.02
	i += 0.02

print("x_x1: ", x_x1)
print("x_x2: ", x_x2)
blues = plt.plot(x_x1, x_x2, 'bo')
reds = plt.plot(z_x1, z_x2, 'ro')
plt.plot([-1, 1], [0, 0], 'go')
plt.show()