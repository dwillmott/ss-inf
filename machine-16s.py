import keras
import numpy as np
import time
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

datafile = 'data/crw16s-filtered.txt'
length = 1828
h1size = 60
weight = k.constant(200.0)

plt.gray()

# ------------------------

def SelfCartesian(x):
    x_expanded = k.expand_dims(x, axis = -2)
    x_tiled = k.repeat_elements(x_expanded, length//4, axis=-2)
    x_transposed = k.permute_dimensions(x_tiled, (0,2,1,3))
    x_concat = k.concatenate([x_tiled, x_transposed], axis=-1)
    #print(k.int_shape(x), k.int_shape(x_expanded), k.int_shape(x_tiled), k.int_shape(x_concat))
    return x_concat


def SelfCartesianShape(input_shape):
    shape = list(input_shape)
    return [shape[0], shape[1], shape[1], shape[2]*2]


def weighted_cross_entropy(target, predicted):
    
    class_weights = k.cast(k.argmax(target, axis = -1), dtype = k.floatx())
    class_weights = class_weights*weight + (1 - class_weights)
    unweighted_losses = k.categorical_crossentropy(target=target, output=predicted)
    weighted_losses = unweighted_losses * class_weights * k.sum(target, axis=-1)
    
    loss = k.mean(weighted_losses)
    return loss


# ------------------------


def plotresults(batch_x, batch_y, batch_yhat, i):
    
    seqlengths = np.argmin(np.sum(batch_x, axis=2), axis=1)
    seqlengths = [seqlength if seqlength > 1 else batch_x.shape[1] for seqlength in seqlengths]
    norm = colors.Normalize(vmin=0., vmax=1.)
    
    
    #fig, axes = plt.subplots(1,3)
    seqlength = seqlengths[0]
    #axes[0].imshow(batch_y[0,:seqlength,:seqlength,0], norm = norm, interpolation='nearest')
    #axes[1].imshow(np.triu(batch_yhat[0,:seqlength,:seqlength,0]), norm = norm, interpolation='nearest')
    #axes[2].imshow(np.triu(batch_yhat[0,:seqlength,:seqlength,0]>0.5), norm = norm, interpolation='nearest')
    #fig.savefig("ss-16s.png", dpi=300)
    #plt.close(fig)
    
    
    #examples = 2
    #fig, axes = plt.subplots(examples,3)
    #for k, seqlength in enumerate(seqlengths[:examples]):
        
        #axes[k,0].imshow(batch_y[k,:seqlength,:seqlength,0], norm = norm, interpolation='nearest')
        #axes[k,1].imshow(np.triu(batch_yhat[k,:seqlength,:seqlength,0]), norm = norm, interpolation='nearest')
        #axes[k,2].imshow(np.triu(batch_yhat[k,:seqlength,:seqlength,0]>0.5), norm = norm, interpolation='nearest')
    #fig.savefig("ss-16s.png", dpi=300)
    #plt.close(fig)
    
    fig = plt.figure()
    #ax = plt.Axes(fig)
    plt.imshow(batch_y[0,:seqlength,:seqlength,0], norm = norm, interpolation='nearest')
    fig.savefig("ss-16s-truth.png", dpi=300)
    plt.close(fig)
    
    fig = plt.figure()
    plt.imshow(np.triu(batch_yhat[0,:seqlength,:seqlength,0]), norm = norm, interpolation='nearest')
    fig.savefig("ss-16s-pred.png", dpi=400)
    plt.close(fig)
    
    return

def plotlosses(losses):
    
    fig = plt.figure()
    plt.plot(losses)
    fig.savefig("ss-16s-losses.png")
    plt.close(fig)
    
    return
    
    #seqlengths = np.argmin(np.sum(batch_x, axis=2), axis=1)
    #seqlengths = [seqlength if seqlength > 1 else batch_x.shape[1] for seqlength in seqlengths]
    #norm = colors.Normalize(vmin=0., vmax=1.)
    
    
    ##fig, axes = plt.subplots(1,3)
    #seqlength = seqlengths[0]
    ##axes[0].imshow(batch_y[0,:seqlength,:seqlength,0], norm = norm, interpolation='nearest')
    ##axes[1].imshow(np.triu(batch_yhat[0,:seqlength,:seqlength,0]), norm = norm, interpolation='nearest')
    ##axes[2].imshow(np.triu(batch_yhat[0,:seqlength,:seqlength,0]>0.5), norm = norm, interpolation='nearest')
    ##fig.savefig("ss-16s.png", dpi=300)
    ##plt.close(fig)
    
    
    ##examples = 2
    ##fig, axes = plt.subplots(examples,3)
    ##for k, seqlength in enumerate(seqlengths[:examples]):
        
        ##axes[k,0].imshow(batch_y[k,:seqlength,:seqlength,0], norm = norm, interpolation='nearest')
        ##axes[k,1].imshow(np.triu(batch_yhat[k,:seqlength,:seqlength,0]), norm = norm, interpolation='nearest')
        ##axes[k,2].imshow(np.triu(batch_yhat[k,:seqlength,:seqlength,0]>0.5), norm = norm, interpolation='nearest')
    ##fig.savefig("ss-16s.png", dpi=300)
    ##plt.close(fig)
    
    #fig = plt.figure()
    ##ax = plt.Axes(fig)
    #plt.imshow(batch_y[0,:seqlength,:seqlength,0], norm = norm, interpolation='nearest')
    #fig.savefig("ss-16s-truth.png", dpi=300)
    #plt.close(fig)
    
    #fig = plt.figure()
    #plt.imshow(np.triu(batch_yhat[0,:seqlength,:seqlength,0]), norm = norm, interpolation='nearest')
    #fig.savefig("ss-16s-pred.png", dpi=400)
    #plt.close(fig)
    
    #return

# ------------------------




#if length == None:
    #length = findsize(datafile)

#print(length)

model = Sequential()
model.add(Conv1D(filters=10, kernel_size=9, strides=4, activation='relu', padding='same', input_shape = (length, 5)))
model.add(Bidirectional(LSTM(h1size, return_sequences = True)))
model.add(Lambda(SelfCartesian, output_shape = SelfCartesianShape))
model.add(Conv2D(filters=20, kernel_size=5, activation='relu', padding='same'))
model.add(Conv2DTranspose(filters=20, kernel_size=4, strides=2, activation='relu', padding='same'))
model.add(Conv2DTranspose(filters=2, kernel_size=4, strides=2, activation='relu', padding='same'))
#model.add(Conv2D(filters=2, kernel_size=5, padding='same'))
model.add(Activation('softmax'))

opt = Adam(lr=0.0001)
model.compile(optimizer=opt,
              loss = 'categorical_crossentropy',
              metrics=['accuracy'],
              sample_weight_mode = "temporal")

print(model.summary())

batchsize = 16



batch_generator = batch_generator(datafile, batchsize, length)
testbatch = makebatch(datafile, batchsize)

losses = []
# training loop
for i in range(5000):
    t = time.time()
    batch_x, batch_y, lengths = next(batch_generator)
    t2 = time.time()
    loss = model.train_on_batch(batch_x, batch_y)
    batch_yhat = model.predict_on_batch(batch_x)
    
    plotresults(batch_x, batch_y, batch_yhat, i)
    
    losses.append(loss[0])
    plotlosses(losses)
    
    difference = batch_yhat - batch_y
    true = batch_y*(batch_yhat > 0.5)
    false = batch_y*(batch_yhat < 0.5)
    tn = np.sum(true[:,:,:,0], axis = (1,2))
    tp = np.sum(true[:,:,:,1], axis = (1,2))
    fn = np.sum(false[:,:,:,1], axis = (1,2))
    fp = np.sum(false[:,:,:,0], axis = (1,2))
    accuracyarray = np.stack([tn, tp, fn, fp])
    
    metrics = getaccuracy(batch_x, batch_y, batch_yhat)
    
    
    print('{:4d}, {:5.5f}, {:5.3f}, {:5.3f}'.format(i, loss[0], t2-t, time.time()-t), 'tp: {:3.0f}/{:3.0f} ({:5.1f}%)  fp: {:5.0f}/{:5.0f} ({:5.1f}%)'.format(accuracyarray[1,0], accuracyarray[1,0]+accuracyarray[2,0], 100.0*accuracyarray[1,0] / (accuracyarray[1,0]+accuracyarray[2,0]), accuracyarray[3,0], accuracyarray[0,0]+accuracyarray[3,0], 100.0*accuracyarray[3,0] / (accuracyarray[0,0]+accuracyarray[3,0])), end='')
    
    print(' PPV: %0.3f  sen: %0.3f  acc: %0.3f' % (metrics[:3]))
    

