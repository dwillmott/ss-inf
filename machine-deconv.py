import keras
import numpy as np
import time
import sys
from matplotlib import colors
import matplotlib.pyplot as plt
import keras.backend as k

from keras.models import Sequential
from keras.layers import Dense, LSTM, Lambda, Conv1D, Conv2D, Conv2DTranspose, Activation, Bidirectional
from keras.optimizers import RMSprop, Adam
from keras.regularizers import l2

#from custom_layers import *
from makebatches import *
from custom import *

#from PIL import Image

np.set_printoptions(linewidth = 300, precision = 5, suppress = True)

datafile = 'data/crw5s-comparative.txt'
length = 250
h1size = 100
batchsize = 16
try:
    weight = k.constant(float(sys.argv[1]))
except:
    weight = 50
convfactor = 2

plt.gray()

# ------------------------

def SelfCartesian(x):
    x_expanded = k.expand_dims(x, axis = -2)
    x_tiled = k.repeat_elements(x_expanded, length//convfactor, axis=-2)
    x_transposed = k.permute_dimensions(x_tiled, (0,2,1,3))
    x_concat = k.concatenate([x_tiled, x_transposed], axis=-1)
    #print(k.int_shape(x), k.int_shape(x_expanded), k.int_shape(x_tiled), k.int_shape(x_concat))
    return x_concat


def SelfCartesianShape(input_shape):
    shape = list(input_shape)
    return [shape[0], shape[1], shape[1], shape[2]*2]


def weighted_cross_entropy(onehot_labels, logits):
    
    class_weights = k.cast(k.argmax(onehot_labels, axis = -1), dtype = k.floatx())
    class_weights = class_weights*weight + (1 - class_weights)
    unweighted_losses = k.categorical_crossentropy(target=onehot_labels, output=logits)
    weighted_losses = unweighted_losses * class_weights * k.sum(onehot_labels, axis=-1)
    
    loss = k.mean(weighted_losses)
    return loss


layers = [Conv1D(filters=10, kernel_size=11, strides=convfactor, activation='relu', padding='same', input_shape = (length, 5)),
          Bidirectional(LSTM(h1size, return_sequences = True), input_shape = (length, 5)),
          Lambda(SelfCartesian, output_shape = SelfCartesianShape),
          Conv2D(filters=30, kernel_size=11, activation='relu', padding='same'),
          Conv2DTranspose(filters=20, kernel_size=8, strides=convfactor, activation='relu', padding='same'),
          Conv2D(filters=2, kernel_size=5, padding='same'),
          Activation('softmax')]

model = Sequential(layers)
#model.add(Conv1D(filters=10, kernel_size=9, strides=convfactor, activation='relu', padding='same', input_shape = (length, 5)))
#model.add(Bidirectional(LSTM(h1size, return_sequences = True), input_shape = (length, 5)))
#model.add(Lambda(SelfCartesian, output_shape = SelfCartesianShape))
#model.add(Conv2D(filters=30, kernel_size=11, activation='relu', padding='same'))
#model.add(Conv2DTranspose(filters=20, kernel_size=5, strides=convfactor, activation='relu', padding='same'))
#model.add(Conv2D(filters=2, kernel_size=5, padding='same'))
#model.add(Activation('softmax'))

opt = Adam(lr=0.0001)
model.compile(optimizer=opt,
              loss = weighted_cross_entropy,
              metrics=['accuracy'],
              sample_weight_mode = "temporal")

print(model.summary())


batch_generator = batch_generator('data/crw5s-comparative.txt', batchsize, length)
testbatch = makebatch(datafile, batchsize)

# training loop
for i in range(10000):
    t = time.time()
    batch_x, batch_y, batch_lengths = next(batch_generator)
    #length = batch_x.shape[1]
    loss = model.train_on_batch(batch_x, batch_y)
    batch_yhat = model.predict_on_batch(batch_x)
    
    if i % 20 == 0:
        plotresults(batch_x, batch_y, batch_yhat, i)
        
        difference = batch_yhat - batch_y
        true = batch_y*(batch_yhat > 0.5)
        false = batch_y*(batch_yhat < 0.5)
        tn = np.sum(true[:,:,:,0], axis = (1,2))
        tp = np.sum(true[:,:,:,1], axis = (1,2))
        fn = np.sum(false[:,:,:,1], axis = (1,2))
        fp = np.sum(false[:,:,:,0], axis = (1,2))
        accuracyarray = np.stack([tn, tp, fn, fp])
        
        metrics = getaccuracy(batch_x, batch_y, batch_yhat)
        
        print('{:4d}, {:5.5f}, {:5.3f}'.format(i, loss[0], time.time()-t), 'tp: {:2.0f}/{:2.0f} ({:5.1f}%)  fp: {:5.0f}/{:5.0f} ({:5.1f}%)'.format(accuracyarray[1,0], accuracyarray[1,0]+accuracyarray[2,0], 100.0*accuracyarray[1,0] / (accuracyarray[1,0]+accuracyarray[2,0]), accuracyarray[3,0], accuracyarray[0,0]+accuracyarray[3,0], 100.0*accuracyarray[3,0] / (accuracyarray[0,0]+accuracyarray[3,0])))
        #print(metrics)
        #print('     PPV: %0.3f  sen: %0.3f  acc: %0.3f      PPV: %0.3f  sen: %0.3f  acc: %0.3f' % (metrics))
        print('                                                                    PPV: %0.3f  sen: %0.3f  acc: %0.3f' % (metrics[:3]))
    

