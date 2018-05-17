import keras
import numpy as np
import time
import matplotlib.pyplot as plt
import keras.backend as k

from keras.models import Sequential
from keras.layers import Dense, LSTM, Lambda, Conv1D, Conv2D, Activation, Bidirectional
from keras.optimizers import RMSprop, Adam
from keras.regularizers import l2

#from custom_layers import *
from makebatches import *

#from PIL import Image

np.set_printoptions(linewidth = 300, precision = 5, suppress = True)

datafile = 'data/crw5s-comparative.txt'
length = 150
h1size = 40

plt.gray()

# ------------------------

def SelfCartesian(x):
    x_expanded = k.expand_dims(x, axis = -2)
    x_tiled = k.repeat_elements(x_expanded, length, axis=-2)
    x_transposed = k.permute_dimensions(x_tiled, (0,2,1,3))
    x_concat = k.concatenate([x_tiled, x_transposed], axis=-1)
    #print(k.int_shape(x), k.int_shape(x_expanded), k.int_shape(x_tiled), k.int_shape(x_concat))
    return x_concat


def SelfCartesianShape(input_shape):
    shape = list(input_shape)
    return [shape[0], shape[1], shape[1], shape[2]*2]


def weighted_cross_entropy(onehot_labels, logits):
    weight = k.constant(100.0)
    class_weights = k.cast(k.argmax(onehot_labels, axis = -1), dtype = k.floatx())
    class_weights = class_weights*weight + (1 - class_weights)
    unweighted_losses = k.categorical_crossentropy(logits, onehot_labels)
    weighted_losses = unweighted_losses * class_weights * k.sum(onehot_labels, axis=-1)
    
    loss = k.mean(weighted_losses)
    return loss

# ------------------------




#if length == None:
    #length = findsize(datafile)

#print(length)

model = Sequential()
model.add(Bidirectional(LSTM(h1size, return_sequences = True), input_shape = (length, 5)))
model.add(Lambda(SelfCartesian, output_shape = SelfCartesianShape))
model.add(Conv2D(filters=50, kernel_size=7, activation='relu', padding='same'))
model.add(Conv2D(filters=2, kernel_size=5, activation='relu', padding='same'))
model.add(Activation('softmax'))

opt = Adam(lr=0.0001)
model.compile(optimizer=opt,
              loss = weighted_cross_entropy,
              metrics=['accuracy'],
              sample_weight_mode = "temporal")

print(model.summary())

batchsize = 10



batch_generator = batch_generator('data/crw5s-comparative.txt', batchsize, length)

# training loop
for i in range(500):
    print(i)
    t = time.time()
    batch_x, batch_y = next(batch_generator)
    #print(time.time() - t)
    #print(batch_x.shape, batch_y.shape)
    loss = model.train_on_batch(batch_x, batch_y)
    batch_yhat = model.predict_on_batch(batch_x)
    
    plt.subplot(1,2,1)
    plt.imshow(batch_y[0,:,:,0], interpolation='nearest')
    plt.subplot(1,2,2)
    plt.imshow(batch_yhat[0,:,:,0], interpolation='nearest')
    plt.savefig("ss.png", dpi=200)
    
    print(loss[0], loss[1],time.time() - t)
    

