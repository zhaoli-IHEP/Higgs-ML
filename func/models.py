import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import SGD, Adam, Nadam


def our_model(img_rows,img_cols):
	model=Sequential()
	model.add(Conv2D(64,(3,3),padding='valid',kernel_initializer="uniform",input_shape=(img_rows,img_cols,1)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2),strides=2))
	model.add(Dropout(0.25))
	
	model.add(Conv2D(64,(3,3),padding='same',kernel_initializer="uniform"))
	model.add(Activation('relu'))
	model.add(Conv2D(64,(3,3),padding='same',kernel_initializer="uniform"))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2),strides=2))
	model.add(Dropout(0.5))
	
	model.add(Conv2D(128,(3,3),padding='same',kernel_initializer="uniform"))
	model.add(Activation('relu'))
	model.add(Conv2D(128,(3,3),padding='same',kernel_initializer="uniform"))
	model.add(Activation('relu'))
	model.add(Conv2D(128,(3,3),padding='same',kernel_initializer="uniform"))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2),strides=2))
	model.add(Dropout(0.5))
	
	model.add(Conv2D(128,(3,3),padding='same',kernel_initializer="uniform"))
	model.add(Activation('relu'))
	model.add(Conv2D(128,(3,3),padding='same',kernel_initializer="uniform"))
	model.add(Activation('relu'))
	model.add(Conv2D(128,(3,3),padding='same',kernel_initializer="uniform"))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2),strides=2))
	model.add(Dropout(0.5))
	
	model.add(Flatten())
	model.add(Dense(128))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1))
	model.add(Activation('sigmoid'))

	adam = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
	model.compile(loss='binary_crossentropy',optimizer = adam, metrics=['accuracy'])
	return model

