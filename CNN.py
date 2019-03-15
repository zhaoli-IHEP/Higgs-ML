from __future__ import print_function
import keras
from keras.models import load_model 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import numpy as np
from keras.optimizers import SGD, Adam, Nadam
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math
try:
    import tkinter
except:
    import Tkinter as tkinter

class LossHistory(keras.callbacks.Callback):
  def on_train_begin(self,log={}):
    self.losses = {'batch':[], 'epoch':[]}
    self.accuracy = {'batch':[], 'epoch':[]}
    self.val_loss = {'batch':[], 'epoch':[]}
    self.val_acc = {'batch':[], 'epoch':[]}

  def on_batch_end(self, batch, logs={}):
    self.losses['batch'].append(logs.get('loss'))
    self.accuracy['batch'].append(logs.get('acc'))
    self.val_loss['batch'].append(logs.get('val_loss'))
    self.val_acc['batch'].append(logs.get('val_acc'))

  def on_epoch_end(self, batch, logs={}):
    self.losses['epoch'].append(logs.get('loss'))
    self.accuracy['epoch'].append(logs.get('acc'))
    self.val_loss['epoch'].append(logs.get('val_loss'))
    self.val_acc['epoch'].append(logs.get('val_acc'))

  def loss_plot(self, loss_type):
    iters = range(len(self.losses[loss_type]))
    plt.figure()
    # acc
    plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc', lw=2.0)
    # loss
    plt.plot(iters, self.losses[loss_type], 'g', label='train loss', lw=2.0)
    if loss_type == 'epoch':
        # val_acc
        plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc', lw=2.0)
        # val_loss
        plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss', lw=2.0)
    plt.grid(True)
    plt.xlabel(loss_type, fontsize=16)
    plt.ylabel('acc-loss', fontsize=16)
    plt.legend(loc="upper right")
    plt.show()




data1= pd.read_table("/home/ligx/work/Z/ZH/data/l/train.txt",sep=',')
data2= pd.read_table("/home/ligx/work/Z/ZH/data/l/test.txt",sep=',')

Train_number = len(data1)
test_number = len(data2)
total_number = len(data1)+len(data2)

print ("total_number:", total_number)
print ("test_number:", test_number)
print ("Train_number:", Train_number)

A1 = data1.values
B1 = data2.values

np.random.shuffle(A1)
np.random.shuffle(B1)

A2 = A1[:,2:1862]
B2 = B1[:,2:1862] 

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

Train_image = A2.reshape(Train_number,30,62,1)
Train_label = A1[:,1:2]
test_image = B2.reshape(test_number,30,62,1)
test_label = B1[:,1:2]
test_weight = B1[:,0]

np.save('Train_image',Train_image)
np.save('Train_label',Train_label)
np.save('test_image',test_image)
np.save('test_label',test_label)
np.save('test_weight',test_weight)

X_train, X_valid, y_train, y_valid =  train_test_split(Train_image,Train_label,test_size=0.1,random_state=22)

print ("train shape:", X_train.shape)
print ("valid shape:", X_valid.shape)

x_train = X_train.astype('float32')
x_valid = X_valid.astype('float32')

nb_filters=64
batch_size=128
img_rows,img_cols =30,62
pool_size=2
nb_classes=2
nb_epoch=50

input_shape = (img_rows,img_cols,1)

model=Sequential()
model.add(Conv2D(nb_filters,(3,3),padding='valid',kernel_initializer="random_normal",input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(pool_size,pool_size),strides=2))
model.add(Dropout(0.5))

model.add(Conv2D(nb_filters,(3,3),padding='valid',kernel_initializer="random_normal"))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(pool_size,pool_size),strides=2))
model.add(Dropout(0.5))

model.add(Conv2D(nb_filters,(3,3),padding='valid',kernel_initializer="random_normal"))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(pool_size,pool_size),strides=2))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(1))
model.add(Activation('sigmoid'))

adam = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
#nadam = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
#sgd = SGD(lr=0.00005, momentum=0.0, decay=0.0, nesterov=False)
model.compile(loss='binary_crossentropy',optimizer = adam, metrics=['accuracy'])
#model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])
history = LossHistory()

#checkpoint = ModelCheckpoint(filepath="try.hdf5", monitor='val_loss', verbose=1, save_best_only=True, mode='min', period = 1)
early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto')
 					  
model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epoch,
          verbose=1, validation_data=(x_valid, y_valid),callbacks=[early_stopping, history])


TrainPrediction = model.predict_proba(x_train)	
ValidPrediction = model.predict_proba(x_valid)	  
TestPrediction = model.predict_proba(test_image)

model.save('l_P_ZH.h5')

fpr, tpr, thresh = metrics.roc_curve(test_label, TestPrediction, pos_label=None, sample_weight=test_weight, drop_intermediate=True)
auc = metrics.auc(fpr,tpr,reorder=True)
print ('AUC :',auc)


score = model.evaluate(test_image, test_label, verbose=0)
print ('socre :',score)

plt.figure(figsize=(8.5,3.7))
plt.subplot(1,2,1)
plt.plot(fpr,tpr)
plt.xlabel('False Positive Rate',fontsize=16)
plt.ylabel('True Positive Rate',fontsize=16)
plt.xlim(-.05,1)
plt.ylim(0,1.05)
plt.title('ROC',fontsize=16)
plt.show()

z=0
k=0


#elec
#N=2069*tpr+10203*fpr

#muon
#N=2036*tpr+6257*fpr

#lepton
N=25793*tpr+23681*fpr

sqrtN=[math.sqrt(i) for i in N]

#elec
#delta_kg=sqrtN/(2.0*2069*tpr)

#muon
#delta_kg=sqrtN/(2.0*2036*tpr)

#lepton
delta_kg=sqrtN/(2*25793*tpr)



best=min(delta_kg)
#print ('delta_kg :',delta_kg)

for i in delta_kg:
	z+=1
	if best==i:
		k=z
print ('best point number :',k)
print ('best tpr :',tpr[k-1:k])
print ('best fpr :',fpr[k-1:k])
print ('minimal delta_kg :',best)

zz=0
b=0
a=0.001
Figure=[0,0,0,0,0,0,0,0]
for i in tpr:
	zz+=1
	for j in range(b,8):
		if (tpr[zz-1:zz]-0.60-0.05*j) < a and (tpr[zz-1:zz]-0.60-0.05*j) > 0:
			#print 'tpr :',tpr[zz-1:zz], 'fpr :',fpr[zz-1:zz], 'delta_kg :',delta_kg[zz-1:zz]
			if fpr[zz-1:zz]<0.1:
				print (round(float(fpr[zz-1:zz]*100),2), '\% ', sep='', end='')
			else:
				print (round(float(fpr[zz-1:zz]*100),1), '\% ', sep='', end='')
			print ('& ', round(float(delta_kg[zz-1:zz]*100),2), '\% ', sep='')
			Figure[b]=round(float(delta_kg[zz-1:zz]*100),6)			
			b+=1
			break

print (Figure)


plt.figure(figsize=(8.5,3.7))
plt.subplot(1,2,1)
plt.plot(tpr,delta_kg,lw=2.0)
plt.xlabel('True Positive Rate',fontsize=16)
plt.ylabel('delta_kg',fontsize=16)
plt.xlim(0.6,1)
plt.ylim(0.0,0.01)
plt.title('Uncertainty',fontsize=16)
plt.show()

history.loss_plot('epoch')

