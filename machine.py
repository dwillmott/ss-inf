import keras
import numpy as np
import time

from keras.models import Sequential
from keras.layers import Dense, LSTM, Lambda, Conv1D, Conv2D, Activation, Bidirectional
from keras.optimizers import RMSprop
from keras.regularizers import l2

from custom_layers import *
from makebatches import *

#from PIL import Image


np.set_printoptions(linewidth = 300, precision = 5, suppress = True)
length = 135
h1size = 30

model = Sequential()
model.add(Bidirectional(LSTM(h1size, return_sequences = True), input_shape = (length, 6)))
model.add(Lambda(SelfCartesian, output_shape = SelfCartesianShape))
model.add(Conv2D(filters=20, kernel_size=5, activation='relu', padding='same'))
model.add(Conv2D(filters=2, kernel_size=5, activation='relu', padding='same'))
model.add(Activation('softmax'))

opt = RMSprop(lr=0.0001)
model.compile(optimizer=opt,
              loss = weighted_cross_entropy,
              metrics=['accuracy'],
              sample_weight_mode = "temporal")

print(model.summary())

batchsize = 5

batch_generator = batch_generator('data/crw5s.txt', batchsize, length)

# training loop
for i in range(200):
    print(i)
    t = time.time()
    batch_x, batch_y = batch_generator.next()
    model.train_on_batch(batch_x, batch_y)
    batch_yhat = model.predict_on_batch(batch_x)
    
    print(time.time() - t)
    
    #if i % 5 == 0:
        #for j in range(batchsize):
            #print(batch_yhat[j, 40:55, 25:35, 0])
            #thres = np.zeros(batch_yhat.shape)
            #thres[batch_yhat > 0.5] = 1
            #print(thres[j, 40:55, 25:35, 0] - batch_y[j, 40:55, 25:35, 0])
            
            #accura = thres[0,:,:,0] - batch_y[0,:,:,0]
            #print('incorrect predictions: %d' % (np.sum(np.abs(accura)),))
    

