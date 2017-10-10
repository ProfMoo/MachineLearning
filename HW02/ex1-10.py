import numpy as np
import scipy
import matplotlib

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt; plt.rcdefaults()
from random import randint

def flip():
	flip = randint(0, 1)
	if (flip == 1):
		return True
	else:
		return False


def getHisto(valueList):
	cHisto = []
	i = 0
	while (i < 11):
		cHisto.append(0)
		i += 1

	i = 0
	ctempvalue = 0
	while (i < len(valueList)):
		ctempvalue = valueList[i]
		cHisto[ctempvalue] += 1
		i += 1

	return cHisto

c1valueList = []
cminvalueList = []
crandvalueList = []

numExp = 1000

h = 0
while (h < numExp):
	numCoins = 1000
	numFlips = 10

	coinCount = []
	i = 0
	cminvalue = 10
	cminlocation = 0
	while (i < numCoins):
		coinCount.append(0)
		j = 0
		while (j < numFlips):
			flip = randint(0, 1)
			if (flip == 1):
				coinCount[i] += 1
			j += 1
		if (coinCount[i] < cminvalue):
			cminvalue = coinCount[i]
			cminlocation = i
		i += 1

	c1value = coinCount[0]
	crandlocation = randint(0, numCoins-1)
	crandvalue = coinCount[crandlocation]

	c1valueList.append(c1value)
	crandvalueList.append(crandvalue)
	cminvalueList.append(cminvalue)

	#print("c1value: ", c1value)
	#print("cminvalue: ", cminvalue)
	#print("cminlocation: ", cminlocation)
	#print("crandlocation: ", crandlocation)
	#print("crandvalue: ", crandvalue)
	#print(coinCount)
	h += 1

#now, to generate graphs
c1valueHisto = getHisto(c1valueList)
crandvalueHisto = getHisto(crandvalueList)
cminvalueHisto = getHisto(cminvalueList)

i = 0
while (i < 10):
	c1valueHisto[i] *= 10
	crandvalueHisto[i] *= 10
	cminvalueHisto[i] *= 10
	i += 1

#printing results
print("crandvalueHisto: ", crandvalueHisto)
print("c1valueHisto: ", c1valueHisto)
print("cminvalueHisto: ", cminvalueHisto)

print("c1valueList: ", c1valueList)
print("crandvalueList: ", crandvalueList)
print("cminvalueList: ", cminvalueList)

#plotting results
fig = plt.figure()
ax1 = fig.add_subplot(411)
ax2 = fig.add_subplot(412)
ax3 = fig.add_subplot(413)
objects = ('0','1','2','3','4','5','6','7','8','9','10')
y_pos = np.arange(len(objects))

ax1.bar(y_pos, cminvalueHisto, align='center', alpha=0.5)
ax2.bar(y_pos, crandvalueHisto, align='center', alpha=0.5)
ax3.bar(y_pos, c1valueHisto, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
ax1.set_title("C(min) Histogram")
ax2.set_title("C(rand) Histogram")
ax3.set_title("C(1) Histogram")

##get calculation off avg for crand
etaList = []
i = 0
while (i < 11):
	etaList.append(0)
	i += 1

print (etaList)	

etaList[0] = crandvalueHisto[0] + crandvalueHisto[1] + crandvalueHisto[2] + crandvalueHisto[3] + crandvalueHisto[4] + crandvalueHisto[6] + crandvalueHisto[7] + crandvalueHisto[8] + crandvalueHisto[9] + crandvalueHisto[10]
etaList[1] = crandvalueHisto[0] + crandvalueHisto[1] + crandvalueHisto[2] + crandvalueHisto[3] + crandvalueHisto[7] + crandvalueHisto[8] + crandvalueHisto[9] + crandvalueHisto[10]
etaList[2] = crandvalueHisto[0] + crandvalueHisto[1] + crandvalueHisto[2] + crandvalueHisto[8] + crandvalueHisto[9] + crandvalueHisto[10]
etaList[3] = crandvalueHisto[0] + crandvalueHisto[1] + crandvalueHisto[9] + crandvalueHisto[10]
etaList[4] = crandvalueHisto[0] + crandvalueHisto[10]

i = 0
while (i < 5):
	etaList[i] = etaList[i]/10000
	i += 1

print("EtaList: ", etaList)

#builds = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
builds = np.array([0,1,2,3,4,5,6,7,8,9,10])
y_stack = np.row_stack(([2, 1.637, 0.8986, 0.33, 0.0815, 0.0134, 0, 0, 0, 0, 0], [etaList[0],etaList[1],etaList[2],etaList[3],etaList[4],0,0,0,0,0,0])) 

ax4 = fig.add_subplot(414)
ax4.plot(builds, y_stack[0,:], label='Hoeffding Bound', color='c', marker='o')
ax4.plot(builds, y_stack[1,:], label='Calculated Result', color='g', marker='o')
y_pos = np.arange(len(objects))
ax4.bar(y_pos, etaList, align='center', alpha=0.5)
ax4.set_title("Calculated Epsilon")

plt.tight_layout()

plt.show()