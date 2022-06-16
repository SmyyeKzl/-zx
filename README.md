# -zx
12
import keras 
from keras.datasets import fashion_mnist
from keras.layers  import Dense,Activation,Flatten,Conv2D,MaxPooling2D
#from keras import layers
from keras.models import Sequential

from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
(train_X,train_Y),(test_X,test_Y)=fashion_mnist.load_data()
train_X=train_X.reshape(-1,28,28,1)
test_X=test_X.reshape(-1,28,28,1)
train_X=train_X.astype('float32')
test_X=test_X.astype('float32')
train_X=train_X/255
test_X=test_X/255
train_Y_one_hot=to_categorical(train_Y)
test_Y_one_hot=to_categorical(test_Y)
model=Sequential()
model.add(Conv2D(64,(3,3),input_shape=(28,28,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128))
model.add(Dense(10))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(train_X,train_Y_one_hot,batch_size=32,epochs=3)
test_loss,test_acc=model.evaluate(test_X,test_Y_one_hot)
print('Test Loss', test_loss)
print('test accuracy' ,test_acc)

11.01
import imp
from pickletools import optimize
from random import random
from statistics import mode
from tabnanny import verbose
from warnings import filters
from matplotlib.pyplot import axis
import pandas as pd
import numpy as np


dataSet = pd.read_csv(".\data_diagnosis.csv")
dataSet.drop(["id","Unnamed: 32"],axis=1,inplace=True)

dataSet.diagnosis = [1 if each == "M" else 0 for each in dataSet.diagnosis]
y=dataSet.diagnosis.values
x_data=dataSet.drop(["diagnosis"],axis=1)
x_data.astype("uint8")

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x=scaler.fit_transform(x_data)

from tensorflow.keras.utils import to_categorical
Y=to_categorical(y)

from sklearn.model_selection import train_test_split
trainX,testX,trainy,testy=train_test_split(x,Y,test_size=0.2,random_state=42)


trainX=trainX.reshape(trainX.shape[0],testX.shape[1],1)
testX=testX.reshape(testX.shape[0],testX.shape[1],1)

from keras import layers
from keras import Sequential

verbose,epochs,batch_size=0,10,8
n_features,n_outputs=trainX.shape[1],trainy.shape[1]

model= Sequential()
input_shape=(trainX.shape[1],1)
model.add(layers.Conv1D(filters=8,kernel_size=5,activation='relu',input_shape=input_shape))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling1D(pool_size=3))
model.add(layers.Conv1D(filters=16,kernel_size=5,activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling1D(pool_size=2))
model.add(layers.Flatten())
model.add(layers.Dense(200,activation='relu'))
model.add(layers.Dense(n_outputs,activation='softmax'))
model.summary()
print('başladı')

import keras 
import tensorflow
#model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.compile(loss='binary_crossentropy',
                optimizer=tensorflow.keras.optimizers.Adam(),
                metrics=['accuracy'])  # 编译
dataSet.info()
model.fit(trainX,trainy,epochs=epochs,verbose=1)
_,accuracy=model.evaluate(testX,testy,verbose=0)

print(accuracy)


11

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import keras.optimizers import *
import tensorflow

data =pd.read_csv("data_diagnosis.csv")
data.drop(["id","Unnamed: 32"],axis=1,inplace=True)
data.diagnosis=[1 if each=="M" else 0 for each in data.diagnosis]
y=data.diagnosis.values
x_data=data.drop(["diagnosis"],axis=1)
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x=scaler.fit_transform(x_data)
from tensorflow.keras.utils import to_categorical
Y=to_categorical(y,dtype="uint8")
from sklearn.model_selection import train_test_split
trainX,testX,trainy,testy=train_test_split(x,Y,test_size=0.2,random_state=42)
trainX=trainX.reshape(trainX.shape[0],trainX.shape[1],1)
testX=testX.resahpe(testX.shape[0],testX.shape[1],1)
from keras import layers
from keras.models import Sequential
from keras.layers import Dense
verbose,epochs,batch_size=0,10,8
n_features,n_outputs=trainX.shape[1],trainy.shape[1]
model=Sequential()
input_shape=(trainX.shape[1],1)
model.add(layers.Conv1D(filters=8,kernel_size=5,activation='relu',input_shape=input_shape))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling1D(pool_size=2))
model.add(layers.Conv1D(filters=16,kernel_size=5,activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling1D(pool_size=2))
model.add(layers.Flatten())
model.add(layers.Dense(200,activation='relu'))
model.add(layers.Dense(n_outputs,activation='softmax'))
model.summary()
print("basladı")
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(trainX,trainy,epochs=epochs,batch_size=batch_size,verbose=1)
_,accuracy =model.evaluate(testX,testy,verbose=0)
print(accuracy)


10
from pickletools import optimize
import numpy as np
dataset=np.loadtxt('diabetes1.csv',delimiter=',')
from sklearn.model_selection import train_test_split
training_set_x,test_set_x,training_set_y,test_set_y=train_test_split(dataset[:,:8],dataset[:,8],test_size=0.2)
from keras.models import Sequential
from keras.layers import Dense
model=Sequential()
model.add(Dense(50,input_dim=8,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(training_set_x,training_set_y,epochs=10,batch_size=8)
test_result=model.evaluate(test_set_x,test_set_y)
print("test loss,test acc:",test_result)
