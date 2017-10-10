import matplotlib.pyplot as plt
import numpy as np

builds = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
y_stack = np.row_stack(([0.6875, 0.6875, 0.21875, 0.21875, 0.03125, 0, 0, 0, 0, 0, 0], [0,0,0,0,0,0,0,0,0,0,0])) 

fig = plt.figure(figsize=(11,8))
ax1 = fig.add_subplot(111)

ax1.plot(builds, y_stack[0,:], label='Calculated Result', color='c', marker='o')
ax1.plot(builds, y_stack[1,:], label='Hoeffding Bound', color='g', marker='o')

plt.xticks(builds)
plt.xlabel('Probability')

handles, labels = ax1.get_legend_handles_labels()
lgd = ax1.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5,1))
ax1.grid('on')

plt.show()