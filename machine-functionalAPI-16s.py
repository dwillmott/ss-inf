import keras
import numpy as np
import time
import sys
from sklearn.metrics import confusion_matrix
from matplotlib import colors
import matplotlib.pyplot as plt
import keras.backend as k

from keras.models import Model
from keras.layers import Input, Dense, LSTM, Lambda, Conv1D, Conv2D, Conv2DTranspose, Activation, Bidirectional, concatenate
from keras.optimizers import RMSprop, Adam
from keras.regularizers import l2

#from custom_layers import *
from makebatches import *
from custom import *

#from PIL import Image

np.set_printoptions(linewidth = 300, precision = 5, suppress = True)

datafile = 'data/crw16s-filtered.txt'
length = 1828
h1size = 30
batchsize = 10
weight = k.constant(2000.0)

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

def weighted_binary_cross_entropy(onehot_labels, logits_ex):
    
    logits = k.squeeze(logits_ex, -1)
    labels = onehot_labels[...,1]
    #labels = k.expand_dims(labels, -1)
    class_weights = labels*weight + (1 - labels)
    unweighted_losses = k.binary_crossentropy(target=labels, output=logits)
    weighted_losses = unweighted_losses * class_weights * k.sum(onehot_labels, axis=-1)#, keepdims=True)
    
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
    fig.savefig("ss-16s-truth.png", dpi=400)
    plt.close(fig)
    
    fig = plt.figure()
    plt.imshow(np.triu(1 - batch_yhat[0,:seqlength,:seqlength]), norm = norm, interpolation='nearest')
    fig.savefig("ss-16s-pred.png", dpi=400)
    plt.close(fig)
    
    return

def plotlosses(losses, testlosses=None):
    
    fig = plt.figure()
    plt.plot(losses)
    if testlosses:
        plt.plot(testlosses)
    fig.savefig("ss-5s-losses.png")
    plt.close(fig)
    
    return

def printoutputs(batch_x, batch_preds, batch_lengths, i, loss, t, t2):
    
    confs = np.stack([confusion_matrix(batch_y[k, :batch_lengths[k], :batch_lengths[k], 1].flatten(),
                                batch_preds[k, :batch_lengths[k], :batch_lengths[k]].flatten(),
                                labels=[0,1]).ravel() for k in range(batchsize)])
    
    tn, fp, fn, tp = np.sum(confs, axis=0)
    print('{:4d}, {:5.5f}, {:3.1f}, {:3.1f}'.format(i, loss, t2-t, time.time()-t), end='')
    print('   tn {:8d}, fp {:8d}, fn {:4d}, tp {:4d}'.format(tn, fp, fn, tp), end='')
    tn, fp, fn, tp = np.sum(confs, axis=0) / np.sum(confs)
    print('   tn {:5.5f}, fp {:5.5f}, fn {:5.5f}, tp {:5.5f}'.format(tn, fp, fn, tp))
    
    return

inputs = Input(shape=(length, 5))
input_conv = Conv1D(filters=20, kernel_size=9, strides=4, activation='relu', padding='same')(inputs)
h1 = Bidirectional(LSTM(h1size, return_sequences = True))(input_conv)
h1square = Lambda(SelfCartesian, output_shape = SelfCartesianShape)(h1)
h2square_1 = Conv2D(filters=20, kernel_size=11, activation='relu', padding='same')(h1square)
h2square_2 = Conv2D(filters=20, kernel_size=7, activation='relu', padding='same')(h1square)
h2square_3 = Conv2D(filters=20, kernel_size=3, activation='relu', padding='same')(h1square)
h2square = concatenate([h2square_1, h2square_2, h2square_3], axis = -1)
h3square = Conv2D(filters=20, kernel_size=3, activation='relu', padding='same')(h2square)
h4square = Conv2DTranspose(filters=20, kernel_size=4, strides=2, activation='relu', padding='same')(h3square)
#sequencesquare = Lambda(SelfCartesian, output_shape = SelfCartesianShape)(inputs)
#h3square = concatenate([h3square_1, sequencesquare], axis = -1)
output = Conv2DTranspose(filters=1, kernel_size=4, strides=2, activation='relu', padding='same')(h4square)

opt = Adam(lr=0.0001)
model = Model(input = inputs, output = output)
model.compile(optimizer=opt,
              loss = weighted_binary_cross_entropy)

print(model.summary())


batch_generator = batch_generator(datafile, batchsize, length)
test_x, test_y, test_lengths = makebatch(datafile, batchsize, maxlength = length)

losses = []
testlosses = []
# training loop
for i in range(10000):
    t = time.time()
    batch_x, batch_y, batch_lengths = next(batch_generator)
    t2 = time.time()
    loss = model.train_on_batch(batch_x, batch_y)
    losses.append(loss)
    
    testloss = model.evaluate(test_x, test_y, verbose = 0)
    testlosses.append(testloss)
    
    #batch_yhat = np.squeeze(model.predict_on_batch(batch_x))
    #batch_preds = np.rint(batch_yhat)

    #plotresults(batch_x, batch_y, batch_yhat, i)
    
    #printoutputs(batch_x, batch_preds, batch_lengths, i, loss, t, t2)
    
    #t = time.time()
    #t2 = time.time()
    
    if i % 3 == 0:
    
        test_yhat = np.squeeze(model.predict_on_batch(test_x))
        test_preds = np.rint(test_yhat)
        
        plotlosses(losses, testlosses)
        plotresults(test_x, test_y, test_yhat, i)
        
        printoutputs(test_x, test_preds, test_lengths, i, testloss, t, t2)
    
    
    
    #confs = np.stack([confusion_matrix(batch_y[k, :batch_lengths[k], :batch_lengths[k], 1].flatten(),
                                #batch_preds[k, :batch_lengths[k], :batch_lengths[k]].flatten(),
                                #labels=[0,1]).ravel() for k in range(batchsize)])
    
    #tn, fp, fn, tp = np.sum(confs, axis=0)
    #print('{:4d}, {:5.5f}, {:5.3f}, {:5.3f}'.format(i, loss[0], t2-t, time.time()-t), end='')
    #print('   tn {:7d}, fp {:7d}, fn {:4d}, tp {:4d}'.format(tn, fp, fn, tp), end='')
    #tn, fp, fn, tp = np.sum(confs, axis=0) / np.sum(confs)
    #print('   tn {:5.5f}, fp {:5.5f}, fn {:5.5f}, tp {:5.5f}'.format(tn, fp, fn, tp))
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
    

