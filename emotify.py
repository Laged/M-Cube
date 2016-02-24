from glob import glob
import struct

import keras
import numpy as np
import array

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense, Masking
from keras.layers.recurrent import LSTM
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

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

X = X[:, :200, :]

model = Sequential()

# model.add(BatchNormalization(mode=0))
model.add(Masking(mask_value=0.0, input_shape=(X.shape[1], X.shape[2])))
model.add(TimeDistributedDense(100, init='glorot_uniform', activation=LeakyReLU(alpha=0.1)))

model.add(LSTM(output_dim=100, activation=LeakyReLU(alpha=0.1), inner_activation='hard_sigmoid', return_sequences=True))
model.add(LSTM(output_dim=100, activation=LeakyReLU(alpha=0.1), inner_activation='hard_sigmoid'))

model.add(BatchNormalization(mode=0))
model.add(Dense(2))

print 'compiling'
model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(X, y, batch_size=8, nb_epoch=200, verbose=1)
#score = model.evaluate(X_test, Y_test, batch_size=16)

'''
out: text fileen per rivi
kesto sekunteina, arvo1, arvo2

silentEnergetic_negativePositive_sekuntit

# alaviivat kivoja t matti
'''


