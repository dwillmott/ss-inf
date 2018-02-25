import keras
import numpy as np
import keras.backend as k
import time

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, LSTM, RepeatVector, Lambda, Reshape, Permute, Conv2D, Activation, Bidirectional, Reshape
from keras.optimizers import RMSprop, Adam
from keras.regularizers import l2

from custom_layers import *
from makebatches import *

#from PIL import Image


np.set_printoptions(linewidth = 300, precision = 4, suppress = True)
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
              loss = 'categorical_crossentropy',
              metrics=['accuracy'],
              sample_weight_mode = "temporal")

print(model.summary())

batchsize = 5

batch_generator = batch_generator('data/crw5s.txt', batchsize, length)


#print(batch_x.shape, batch_y.shape)

#sample_weight = (batch_y*10)[:,:,:,0].reshape((batchsize, length**2))
#print(sample_weight[3,40:50, 25:35])
#sample_weight = sample_weight + np.ones(sample_weight.shape)*0.001

#sample_weight = np.ones((batchsize,135**2))


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
    
    #thres = np.zeros(pred.shape)
    
    #thres[pred > 0.5] = 1
    ##print(thres[0, 40:55, 25:35, 0])
    #accura = thres[0,:,:,0] - target[0,:,:,0]
    #print('incorrect predictions: ', np.sum(np.abs(accura)))
    #print()
    #im = Image.fromarray((pred[0,:,:,0]*255).astype()
    #im.save('outputpicture.png')
    #activations = get_activations(model, inputdata)
    #print(activations[0])
    

