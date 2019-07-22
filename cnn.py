from __future__ import print_function
import os, sys
import math

import pandas as pd
import numpy as np

import keras
from keras.models import load_model 
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn import metrics

from func.figure import LossHistory, ROC_plot, deltaKg_plot
from func.models import our_model
from func.file import removedir

try:
    import tkinter
except:
    import Tkinter as tkinter


########################################################## load data ##########################################################

dirs=['npydata','model','plot']
for d in dirs:
	if os.path.exists(d):
		removedir(d)
	os.makedirs(d)

img_rows,img_cols =34,66

data1= pd.read_table('data/train.txt', header=None, sep=',')
data2= pd.read_table('data/test.txt', header=None, sep=',')

Train_number = len(data1)
test_number = len(data2)
total_number = len(data1)+len(data2)

print ('total_number:', total_number)
print ('test_number:', test_number)
print ('Train_number:', Train_number)

A1 = data1.values
B1 = data2.values

np.random.shuffle(A1)
np.random.shuffle(B1)

A2 = A1[:,2:img_rows*img_cols+2]
B2 = B1[:,2:img_rows*img_cols+2] 

#A2_sum=np.sum(A2, axis = 1)
#A2 = A2.T
#A2 /= (A2_sum+10e-8)
#A2 = A2.T
#A2 -= np.mean(A2, axis = 0)
#A2 /= (np.std(A2, axis = 0)+10e-5)

#B2_sum=np.sum(B2, axis = 1)
#B2 = B2.T
#B2 /= (B2_sum+10e-8)
#B2 = B2.T
#B2 -= np.mean(B2, axis = 0)
#B2 /= (np.std(B2, axis = 0)+10e-5)

Train_image = A2.reshape(Train_number,img_rows,img_cols,1)
Train_label = A1[:,1:2]
Train_weight = A1[:,0:1]
test_image = B2.reshape(test_number,img_rows,img_cols,1)
test_label = B1[:,1:2]
test_weight = B1[:,0:1]

#np.save('npydata/Train_image',Train_image)
#np.save('npydata/Train_label',Train_label)
#np.save('npydata/Train_weight',Train_weight)
#np.save('npydata/test_image',test_image)
#np.save('npydata/test_label',test_label)
#np.save('npydata/test_weight',test_weight)

X_train, X_valid, y_train, y_valid =  train_test_split(Train_image,Train_label,test_size=0.1,random_state=22)

print ('train shape:', X_train.shape)
print ('valid shape:', X_valid.shape)

x_train = X_train.astype('float32')
x_valid = X_valid.astype('float32')

############################################## train ######################################################################################################

model=our_model(img_rows,img_cols)

history = LossHistory()
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto') 	
saveBestModel = ModelCheckpoint(filepath='model/best.h5', monitor='val_loss', verbose=0, save_best_only=True, mode='min')			  
		  
model.fit(x_train, y_train, batch_size=128, epochs=100, verbose=1, validation_data=(x_valid, y_valid),callbacks=[early_stopping, saveBestModel, history])

model.save('model/final.h5')

############################################# evaluate ####################################################################################################
  
TestPrediction = model.predict_proba(test_image)

fpr, tpr, thresh = metrics.roc_curve(test_label, TestPrediction, pos_label=None, sample_weight=test_weight, drop_intermediate=True)
auc = metrics.auc(fpr, tpr, reorder=True)
print ('AUC :',auc)

Ng, NB=4090, 21141
delta_kg=[]
for i in range(len(tpr)):
	if tpr[i]==0:
		delta_kg.append(1000)
	else:
		delta_kg.append(math.sqrt(Ng*tpr[i]+NB*fpr[i])/(2.0*Ng*tpr[i]))

best=min(delta_kg)
min_index=delta_kg.index(best)
print ('best point: (tpr, fpr) = (',tpr[min_index],',',fpr[min_index],')')
print ('minimal delta_kg =',best)		

history.loss_plot('epoch')
ROC_plot(tpr, fpr)
deltaKg_plot(tpr, delta_kg)

