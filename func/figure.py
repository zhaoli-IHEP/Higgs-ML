from __future__ import print_function
import keras
import numpy as np
import matplotlib.pyplot as plt
import math


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
    		#val_predict=(np.asarray(self.model.predict(self.validation_data[0]))).round()
    		#val_targ=self.validation_data[1]
    		#out=np.hstack((val_targ, val_predict))
    		#TP,TN,FP,FN=0,0,0,0
    		#for i in range(len(out)):
		#print (round(float(out[order-1:order,0]),8), round(float(out[order-1:order,1]),8))
		#	if out[i,0]==1 and out[i,1]==1:
		#		TP+=1
		#	if out[i,0]==0 and out[i,1]==0:
		#		TN+=1
		#	if out[i,0]==0 and out[i,1]==1:
		#		FP+=1
		#	if out[i,0]==1 and out[i,1]==0:
		#		FN+=1
    		#if TP!=0 and TN!=0 and FP!=0 and FN!=0:
        	#	precision=TP/float(TP+FP);
        	#	recall=TP/float(TP+FN);
        	#	f1=2*precision*recall/(precision+recall);
        	#	print("TP:",TP," TN:",TN," FP:",FP," FN:",FN,"   tpr:",TP/float(TP+FN)," fpr:",FP/float(FP+TN),"	precision:",precision," recall:",recall," f1:",f1)

  	def loss_plot(self, loss_type):
    		iters = range(len(self.losses[loss_type]))
    		plt.figure()
    		plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc', lw=2.0)
    		plt.plot(iters, self.losses[loss_type], 'g', label='train loss', lw=2.0)
    		if loss_type == 'epoch':
        		plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc', lw=2.0)
        		plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss', lw=2.0)
    		plt.grid(True)
    		plt.xlabel(loss_type, fontsize=16)
    		plt.ylabel('acc-loss', fontsize=16)
    		plt.legend(loc="upper right")
    		plt.savefig('plot/loss_acc.png')
    		#plt.show()



def ROC_plot(tpr, fpr):
    	plt.figure(figsize=(8.5,3.7))
    	plt.subplot(1,2,1)
    	plt.plot(tpr,1-fpr)
    	plt.xlabel('Signal Efficiency',fontsize=16)
    	plt.ylabel('Background Rejection',fontsize=16)
    	plt.xlim(0,1)
    	plt.ylim(0,1)
    	plt.title('ROC',fontsize=16)
    	plt.savefig('plot/ROC.png')
    	#plt.show()
    	return



def deltaKg_plot(tpr, delta_kg):
    	plt.figure(figsize=(8.5,3.7))
    	plt.subplot(1,2,1)
    	plt.plot(tpr,delta_kg,lw=2.0)
    	plt.xlabel('True Positive Rate',fontsize=16)
    	plt.ylabel('delta_kg',fontsize=16)
    	plt.xlim(0.4,0.9)
    	plt.ylim(0.01,0.02)
    	plt.title('Uncertainty',fontsize=16)
    	plt.savefig('plot/delta_Kg.png')
    	#plt.show()
	return


