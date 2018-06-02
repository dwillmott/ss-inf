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

datafile = 'data/crw16s-filtered-long.txt'
length = 300
batchsize = 10
weight = k.constant(200.0)
#triumask = k.constant(np.triu(np.ones([length, length])))

plt.gray()

# ------------------------

def SelfCartesian(x):
    newshape = k.stack([1, 1, k.shape(x)[1], 1])
    
    x_expanded = k.expand_dims(x, axis = -2)
    x_tiled = k.tile(x_expanded, newshape)
    x_transposed = k.permute_dimensions(x_tiled, (0,2,1,3))
    x_concat = k.concatenate([x_tiled, x_transposed], axis=-1)
    return x_concat


def SelfCartesianShape(input_shape):
    shape = list(input_shape)
    return [shape[0], shape[1], shape[1], shape[2]*2]


def weighted_binary_cross_entropy(labels, logits):
    
    triumask = k.constant(np.triu(np.ones([k.int_shape(logits)[1], k.int_shape(logits)[1]])))
    class_weights = labels*weight + (1 - labels)
    unweighted_losses = k.binary_crossentropy(target=labels, output=logits)
    weighted_losses = unweighted_losses * class_weights * triumask
    
    loss = k.mean(weighted_losses)
    return loss


def plotresults(square, name):
    norm = colors.Normalize(vmin=0., vmax=1.)
    
    fig = plt.figure()
    plt.imshow(np.triu(1 - square), norm = norm, interpolation='nearest')
    fig.savefig(name, dpi=400)
    plt.close(fig)
    return


def plotlosses(losses, testlosses=None):
    fig = plt.figure()
    plt.plot(losses)
    if testlosses:
        plt.plot(testlosses)
    fig.savefig("ss-sub-losses.png")
    plt.close(fig)
    return

def printoutputs(batch_y, batch_preds, i, loss, t, t2):
    
    confs = np.stack([confusion_matrix(y[y.shape[1]].flatten(),
                                       pred[pred.shape[1]].flatten(),
                                       labels=[0,1]).ravel() for y, pred in zip(batch_y, batch_preds)])
    
    tn, fp, fn, tp = np.sum(confs, axis=0)
    print('{:4d}, {:5.5f}, {:3.1f}, {:4.1f}, {:4.1f}, {:4.1f}'.format(i, loss, t2-t, t3-t2, t4-t3, time.time()-t4), end='')
    print('   tn {:8d}, fp {:8d}, fn {:4d}, tp {:4d}'.format(tn, fp, fn, tp))
    
    return

inputs = Input(shape=(None, 5))

h1 = Bidirectional(LSTM(50, return_sequences = True))(inputs)

h1square = Lambda(SelfCartesian, output_shape = SelfCartesianShape)(h1)

h2square_1 = Conv2D(filters=20, kernel_size=15, activation='relu', padding='same')(h1square)
h2square_2 = Conv2D(filters=20, kernel_size=9, activation='relu', padding='same')(h1square)
h2square_3 = Conv2D(filters=20, kernel_size=5, activation='relu', padding='same')(h1square)
h2square = concatenate([h2square_1, h2square_2, h2square_3], axis=-1)

h3square_1 = Conv2D(filters=20, kernel_size=9, activation='relu', padding='same')(h2square)
h3square_2 = Conv2D(filters=20, kernel_size=5, activation='relu', padding='same')(h2square)
h3square = concatenate([h3square_1, h3square_2], axis=-1)

h4square_1 = Conv2D(filters=20, kernel_size=5, activation='relu', padding='same')(h3square)
sequencesquare = Lambda(SelfCartesian, output_shape = SelfCartesianShape)(inputs)
h4square = concatenate([h4square_1, sequencesquare], axis = -1)

output = Conv2D(filters=1, kernel_size=3, activation='sigmoid', padding='same')(h4square)

opt = Adam(lr=0.0001)
model = Model(input = inputs, output = output)
model.compile(optimizer=opt,
              loss = weighted_binary_cross_entropy)

print(model.summary())


batch_generator = batch_sub_generator(datafile, batchsize, length)
valid_x, valid_y = makebatch_sub(datafile, batchsize, length)


losses = []
testlosses = []
# training loop
for i in range(100):
    t = time.time()
    batch_x, batch_y = next(batch_generator)
    batch_stateseq = np.sum(batch_y, axis = -1)
    t2 = time.time()
    loss = model.train_on_batch(batch_x, np.expand_dims(batch_y, -1))
    losses.append(loss)
    t3 = time.time()
    testloss = model.evaluate(valid_x, np.expand_dims(valid_y, -1), verbose = 0)
    testlosses.append(testloss)
    t4 = time.time()
    
    if i % 10 == 0:
    
        valid_yhat = np.squeeze(model.predict_on_batch(valid_x))
        test_preds = np.rint(valid_yhat)
        
        plotlosses(losses, testlosses)
        plotresults(valid_y[0], 'ss-sub-truth.png')
        plotresults(valid_yhat[0], 'ss-sub-pred.png')
        printoutputs(valid_y, test_preds, i, testloss, t, t2)
    
model.save('sub.hdf5')
