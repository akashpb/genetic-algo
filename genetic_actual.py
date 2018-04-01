import keras
import random
import numpy as np
import sklearn
from keras import losses
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras import metrics
import pandas as pd

neurons = 0
hidden_layers = 0
actual_data = []
newdata = []
optimizers = ['adam', 'sgd', 'RMSprop', 'Adagrad', 'Adadelta']
output = []
#print(op)
#print(i)
actual_layers = 0

data = list(open('genetic.csv', 'r'))
for i in data:
    actual_data.append(i.split("\n"))

for i in actual_data:
    for j in i:
        newdata.append(j.split(","))
#print(newdata)
actual_data = []
for i in newdata:
    if len(i)>1:
        actual_data.append(i)
#print(actual_data)

ndata = np.array(actual_data)

floatdata = ndata.astype(np.float)

train_X = (floatdata[:,[0,1,2]])
#print(len(train_X))

train_Y = (floatdata[:,3])
#print(len(train_Y))

#train_X, train_Y, valid_X, valid_Y = train_test_split(train_X, train_Y, test_size = 0.3, random_state = 42)
X_train = train_X
Y_train = train_Y
#Referencing
train_X = X_train[0:2000, :]
train_Y = Y_train[:2000]
valid_X = X_train[2000:2217, :]
valid_Y = Y_train[2000:2217]

def random_generator():
    for x in range(100):
        neurons = random.randint(1,100)
        hidden_layers = random.randint(1, 10)
        i = random.randint(0,4)
        string = "model.add(Dense(units=" + str(neurons) + ",activation='relu'))"
        activation = optimizers[i]
        actual_output = neural_net(neurons, hidden_layers, string, activation)
        output.append(actual_output)
    
    print("Writing to CSV")
    result1 = pd.DataFrame(output, columns=['actual_layers', 'layers', 'neurons', 'optimizers', 'learning rate', 'accuracy'])
    #result1.loc[:, 'Test Images'] = pd.Series(test_id, index=result1.index)
    result1.to_csv('hyperparameters.csv', index=False)
    print("Completed")


def neural_net(neurons, layers, string, activation):
    model = Sequential()
    if layers == 1:
        model.add(Dense(units=32, activation='relu', input_shape=(3,)))
        eval(string)
        model.add(Dense(units=1))
    elif layers >= 2:
        model.add(Dense(units=32, activation='relu', input_shape=(3,)))
        for i in range(layers):
            eval(string)
        model.add(Dense(units=1))
    model.compile(loss=keras.losses.mean_squared_error, optimizer= str(activation), metrics=['accuracy'] )
    model.fit(train_X, train_Y, epochs=1, validation_data=(valid_X, valid_Y))
    actual_layers = layers + 2
    actual_neurons = neurons + 33
    l = list((actual_layers, layers, actual_neurons, activation, model.optimizer.lr, model.metrics[0]))
    return l


random_generator()
