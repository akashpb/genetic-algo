import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

data = csv.reader(open('hyperparameters.csv'))
dataList = list(data)
dataList = dataList[1:]
npdata = np.array(dataList)
train = 0.7
#print(npdata)
X = npdata[:,[0,1,2,4]].astype(np.float)
Y = npdata[:,5].astype(np.float)
optimizers = []
optimizers.extend(npdata[:,3])

#print(X)
#print(Y)
# Find the optimal plan
# 
#print(optimizers)
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(optimizers)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
optimizers_one_hot = onehot_encoder.fit_transform(integer_encoded)
# print(optimizers_one_hot)
clf = svm.SVC(kernel = 'linear')
X = np.column_stack((X,optimizers_one_hot))

for i in range(len(Y)):
    if(Y[i] < 0.5):
        Y[i] = 0
    elif(Y[i] >= 0.5):
        Y[i] = 1

train_X = X[:int(len(X)*train)]
train_Y = Y[:int(len(Y)*train)]
valid_X = X[int(len(X)*train):]
valid_Y = Y[int(len(Y)*train):]
#Find the plane
clf.fit(train_X, train_Y)

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
# t = clf.predict([[10,5]])

acc = clf.score(valid_X, valid_Y)
print(acc)
#plt.plot(10, 5, color = "black")
plt.plot(xx, yy, color = "red")
plt.show()