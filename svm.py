import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
import keras.utils


data = csv.reader(open('hyperparameters.csv'))
dataList = list(data)
dataList = dataList[1:]
npdata = np.array(dataList)
print(npdata)
X = npdata[:,[0,1,2,4]].astype(np.float)
Y = npdata[:,5].astype(np.float)
optimizers = []
optimizers.extend(npdata[:,4])
print(X)
print(Y)
# Find the optimal plane
# 
optimizers_one_hot = keras.utils.to_categorical(optimizers)
clf = svm.SVC(kernel = 'rbf')
X = np.column_stack((X,optimizers_one_hot))

#Find the plane
clf.fit(X, Y)

# You get an array of weights
W = clf.coef_[0]

bias = clf.intercept_[0]

xx = np.linspace(0,12)
yy = -bias/W[1] - xx*W[0]/W[1]

for i in range(len(X)):
    if(Y[i] == 1):
        plt.plot(X[i,0], X[i,1], 'b+', label = "POS")
    else:
        plt.plot(X[i,0], X[i,1], 'rx', label = "NEG")
        
#plt.legend()
#plt.plot(-bias,+bias, color = "yellow")
t = clf.predict([[10,5]])
print(t)
#plt.plot(10, 5, color = "black")
plt.plot(xx, yy, color = "red")
plt.show()
