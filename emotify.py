from glob import glob
import struct

import keras
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense
from keras.layers.recurrent import LSTM
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

print 'loading data'

for f in glob('*/*.ftw'):
	data = open(f).read()
	print "reading file ", f
	#Parse file name to get SE and NP values
	values = f.split("/")[1].split("_")
	SE = values[0]
	NP = values[1].split(".")[0]
	print "with SE ", SE, " and NP ", NP
	#DONE, direct the values to ML
	total_samples = struct.Struct('i').unpack_from(data, offset=0)[0]
	window_samples = struct.Struct('i').unpack_from(data, offset=4)[0]
	
	print total_samples, window_samples
	
	data = data[8:]
	p = struct.Struct('d')
	
	X = np.zeros((1, total_samples/window_samples, window_samples)).astype(np.float32)
	
	# TODO loop dummy and slow
	for i in range(0, total_samples, window_samples):
		
		if i/window_samples == total_samples/window_samples:
			continue
		
		for j in range(window_samples):

#			print i/window_samples, X.shape[1], total_samples/window_samples

			X[0, i/window_samples, j] = p.unpack_from(data, offset = (i+j)*8)[0]
			
			

print X
# TODO tee y matriisi

model = Sequential()

model.add(TimeDistributedDense(100, init='glorot_uniform', activation=LeakyReLU(alpha=0.1), input_shape=(100, 100)))

model.add(BatchNormalization(mode=0))
model.add(LSTM(output_dim=100, activation=LeakyReLU(alpha=0.1), inner_activation='hard_sigmoid'))

model.add(BatchNormalization(mode=0))
model.add(Dense(2))

model.compile(loss='mean_squared_error', optimizer='adam')

#model.fit(X_train, Y_train, batch_size=16, nb_epoch=10)
#score = model.evaluate(X_test, Y_test, batch_size=16)

'''
out: text fileen per rivi
kesto sekunteina, arvo1, arvo2

silentEnergetic_negativePositive_sekuntit

# alaviivat kivoja t matti
'''


