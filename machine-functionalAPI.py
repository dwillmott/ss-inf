import keras
import numpy as np
import time
import sys
from sklearn.metrics import confusion_matrix
from matplotlib import colors
import matplotlib.pyplot as plt
import keras.backend as k

from keras.models import Model
from keras.layers import Input, Dense, LSTM, Lambda, Conv1D, Conv2D, Activation, Bidirectional
from keras.optimizers import RMSprop, Adam
from keras.regularizers import l2

#from custom_layers import *
from makebatches import *
from custom import *

#from PIL import Image

np.set_printoptions(linewidth = 300, precision = 5, suppress = True)

datafile = 'data/crw5s-comparative.txt'
length = 150
h1size = 20
batchsize = 16
weight = k.constant(float(sys.argv[1]))

plt.gray()

# ------------------------

def SelfCartesian(x):
    x_expanded = k.expand_dims(x, axis = -2)
    x_tiled = k.repeat_elements(x_expanded, length, axis=-2)
    x_transposed = k.permute_dimensions(x_tiled, (0,2,1,3))
    x_concat = k.concatenate([x_tiled, x_transposed], axis=-1)
    print(k.int_shape(x), k.int_shape(x_expanded), k.int_shape(x_tiled), k.int_shape(x_concat))
    return x_concat


def SelfCartesianShape(input_shape):
    shape = list(input_shape)
    return [shape[0], shape[1], shape[1], shape[2]*2]

def weighted_binary_cross_entropy(onehot_labels, logits):
    
    labels = onehot_labels[...,1]
    labels = k.expand_dims(labels, -1)
    class_weights = labels*weight + (1 - labels)
    unweighted_losses = k.binary_crossentropy(target=labels, output=logits)
    weighted_losses = unweighted_losses * class_weights * k.sum(onehot_labels, axis=-1, keepdims=True)
    
    loss = k.mean(weighted_losses)
    return loss

def weighted_cross_entropy(onehot_labels, logits):
    
    class_weights = k.cast(k.argmax(onehot_labels, axis = -1), dtype = k.floatx())
    class_weights = class_weights*weight + (1 - class_weights)
    unweighted_losses = k.categorical_crossentropy(target=onehot_labels, output=logits)
    weighted_losses = unweighted_losses * class_weights * k.sum(onehot_labels, axis=-1)
    
    loss = k.mean(weighted_losses)
    return loss

def plotresults(batch_x, batch_y, batch_yhat, i):
    
    seqlengths = np.argmin(np.sum(batch_x, axis=2), axis=1)
    seqlengths = [seqlength if seqlength > 1 else batch_x.shape[1] for seqlength in seqlengths]
    norm = colors.Normalize(vmin=0., vmax=1.)
    
    fig, axes = plt.subplots(4,3)
    for k, seqlength in enumerate(seqlengths[:4]):
        
        
        #print(batch_yhat[k,:seqlength,:seqlength].shape)
        #print((1 - batch_yhat[k,:seqlength,:seqlength]).shape)
        #print(np.triu(1 - batch_yhat[k,:seqlength,:seqlength]).shape)
        
        axes[k,0].imshow(batch_y[k,:seqlength,:seqlength,0], norm = norm, interpolation='nearest')
        axes[k,1].imshow(np.triu(1 - batch_yhat[k,:seqlength,:seqlength]), norm = norm, interpolation='nearest')
        axes[k,2].imshow(np.triu(1 - batch_yhat[k,:seqlength,:seqlength]>0.5), norm = norm, interpolation='nearest')
    fig.savefig("pictureoutput/ss-%05d.png" % (i,), dpi=200)
    plt.close(fig)
    
    return


inputs = Input(shape=(length, 5))
h1 = Bidirectional(LSTM(h1size, return_sequences = True))(inputs)
h1square = Lambda(SelfCartesian, output_shape = SelfCartesianShape)(h1)
h2square = Conv2D(filters=20, kernel_size=11, activation='relu', padding='same')(h1square)
h3square = Conv2D(filters=20, kernel_size=5, activation='relu', padding='same')(h2square)
output_logits = Conv2D(filters=1, kernel_size=5, padding='same')(h3square)
#output = Activation('softmax')(output_logit)

#model.add(Bidirectional(LSTM(h1size, return_sequences = True), input_shape = (length, 5)))
#model.add(Lambda(SelfCartesian, output_shape = SelfCartesianShape))
#model.add(Conv2D(filters=30, kernel_size=11, activation='relu', padding='same'))
#model.add(Conv2D(filters=20, kernel_size=5, activation='relu', padding='same'))
#model.add(Conv2D(filters=2, kernel_size=5, padding='same'))
#model.add(Activation('softmax'))

opt = Adam(lr=0.0001)
model = Model(input = inputs, output = output_logits)
model.compile(optimizer=opt,
              loss = weighted_binary_cross_entropy,
              metrics=['accuracy'],
              sample_weight_mode = "temporal")

print(model.summary())


batch_generator = batch_generator('data/crw5s-comparative.txt', batchsize, length)
#testbatch = makebatch(datafile, batchsize)

# training loop
for i in range(10000):
    t = time.time()
    batch_x, batch_y, batch_lengths = next(batch_generator)
    t2 = time.time()
    loss = model.train_on_batch(batch_x, batch_y)
    batch_yhat = np.squeeze(model.predict_on_batch(batch_x))
    batch_yhat = 1 / (1 + np.exp(-batch_yhat))
    batch_preds = np.rint(batch_yhat)
    
    
        
    print('{:4d}, {:5.5f}, {:5.3f}, {:5.3f}'.format(i, loss[0], t2-t, time.time()-t))
    plotresults(batch_x, batch_y, batch_yhat, i)
    
    confs = np.stack([confusion_matrix(batch_y[k, :batch_lengths[k], :batch_lengths[k], 1].flatten(),
                                batch_preds[k, :batch_lengths[k], :batch_lengths[k]].flatten()).ravel() for k in range(batchsize)])
    
    tn, fp, fn, tp = np.sum(confs, axis=0)
    print('{:4d}, {:5.5f}, {:5.3f}, {:5.3f}'.format(i, loss[0], t2-t, time.time()-t), tn, fp, fn, tp)
    
    #true = batch_y*(batch_yhat > 0.5)
    #false = batch_y*(batch_yhat < 0.5)
    #tn = np.sum(true[:,:,:,0], axis = (1,2))
    #tp = np.sum(true[:,:,:,1], axis = (1,2))
    #fn = np.sum(false[:,:,:,1], axis = (1,2))
    #fp = np.sum(false[:,:,:,0], axis = (1,2))
    #accuracyarray = np.stack([tn, tp, fn, fp])
    
    #metrics = getaccuracy(batch_x, batch_y, batch_yhat)
    
    #print('{:4d}, {:5.5f}, {:5.3f}, {:5.3f}'.format(i, loss[0], t2-t, time.time()-t), 'tp: {:3.0f}/{:3.0f} ({:5.1f}%)  fp: {:5.0f}/{:5.0f} ({:5.1f}%)'.format(accuracyarray[1,0], accuracyarray[1,0]+accuracyarray[2,0], 100.0*accuracyarray[1,0] / (accuracyarray[1,0]+accuracyarray[2,0]), accuracyarray[3,0], accuracyarray[0,0]+accuracyarray[3,0], 100.0*accuracyarray[3,0] / (accuracyarray[0,0]+accuracyarray[3,0])), end='')

    #print(' PPV: %0.3f  sen: %0.3f  acc: %0.3f' % (metrics[:3]))
    

