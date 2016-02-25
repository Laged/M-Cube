from glob import glob
import struct

import keras
import numpy as np
import array
import os

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense, Masking
from keras.layers.recurrent import LSTM
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import model_from_json
from keras.optimizers import Adam

print 'loading data'

X = []
y = []

for f in glob('*/*.ftw'):
	data = open(f).read()
	print "reading file ", f
	#Parse file name to get SE and NP values

	values = f.split("/")[1].split("_")
	SE = float(values[0].replace(',','.'))
	NP = float(values[1].split(".")[0].replace(',','.'))
	print "with SE ", SE, " and NP ", NP

	#DONE, direct the values to ML
	total_samples = struct.Struct('i').unpack_from(data, offset=0)[0]
	window_size = struct.Struct('i').unpack_from(data, offset=4)[0]

	window_count = total_samples/window_size
	
	print "total_samples:", total_samples, "window_size:", window_size, "window_count", window_count
	
	p = array.array('d', data[8:]).tolist()
	Xi = np.array(p).astype(np.float32).reshape((window_count, window_size))
		
	X.append(Xi)
	y.append([SE, NP])

# find max length of X sequences and pad the rest with zeros
maxlen = max(x.shape[0] for x in X)
for i in range(len(X)):
	X[i] = np.concatenate((X[i], np.zeros((maxlen-X[i].shape[0], X[i].shape[1]))), axis=0)

y = np.array(y)
X = np.array(X)

X = X[:, :220 * 5 * 2, :]

# try load model
if os.path.isfile('model.json') and os.path.isfile('model.h5') and False:
	print 'loading model'
	model = model_from_json(open('model.json').read())
	model.load_weights('model.h5')
	print 'compiling'
	model.compile(loss='mean_squared_error', optimizer='adam')

else:
	learningRate = 0.1
	print 'creating model with learningRate ', learningRate
	model = Sequential()

	#model.add(Masking(mask_value=0.0, input_shape=(X.shape[1], X.shape[2])))
	model.add(BatchNormalization(mode=0, input_shape=(X.shape[1], X.shape[2])))
	model.add(TimeDistributedDense(50, init='glorot_uniform', activation='linear')) #, input_shape=(X.shape[1], X.shape[2])))
	model.add(LeakyReLU(alpha=0.1))
	model.add(BatchNormalization(mode=0))
	
	#model.add(LSTM(output_dim=20, activation=LeakyReLU(alpha=0.1), inner_activation='hard_sigmoid', return_sequences=True))
	model.add(LSTM(output_dim=50, activation='tanh', inner_activation='sigmoid'))
	model.add(LeakyReLU(alpha=0.1))
	
	model.add(Dense(20))
	model.add(BatchNormalization(mode=0))

	model.add(LeakyReLU(alpha=0.1))

	model.add(Dense(2))
	adam = Adam(lr=learningRate)#TODO: TRY RMSPROP :_D

	print 'compiling'
	model.compile(loss='mean_squared_error', optimizer=adam)


	#model.load_weights('model.h5')
	model.fit(X, y, batch_size=8, nb_epoch=100, verbose=1)

	# save model
	open('model.json','w').write(model.to_json())
	model.save_weights('model.h5')

#score = model.evaluate(X_test, Y_test, batch_size=16)

'''
out: text fileen per rivi
kesto sekunteina, arvo1, arvo2

silentEnergetic_negativePositive_sekuntit

# alaviivat kivoja t matti
'''


