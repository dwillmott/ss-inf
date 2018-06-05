import keras
import numpy as np
import time
import sys, os
from sklearn.metrics import confusion_matrix
from matplotlib import colors
import matplotlib.pyplot as plt
import keras.backend as k

from keras.models import Model
from keras.layers import Input, Dense, LSTM, Lambda, Conv1D, Conv2D, Conv2DTranspose, Activation, Bidirectional, concatenate, BatchNormalization
from keras.optimizers import RMSprop, Adam
from keras.regularizers import l2

#from custom_layers import *
from makebatches import *
from custom import *
import tensorflow as tf

#from PIL import Image

np.set_printoptions(linewidth = 300, precision = 5, suppress = True)

datafile = 'data/crw16s-filtered-long.txt'
outputdir = 'outputs-bn/'
savename = 'sub-bn'
length = 300
batchsize = 10
weight = k.constant(200.0)
plt.gray()

def weighted_binary_cross_entropy(labels, logits):
    class_weights = labels*weight + (1 - labels)
    unweighted_losses = K.binary_crossentropy(target=labels, output=logits)
    weighted_losses = unweighted_losses * class_weights
    
    loss = K.mean(tf.matrix_band_part(K.squeeze(weighted_losses, -1), 0, -1))
    return loss

loadmodel = False

testsetnames = ['cuniculi', 'vnecatrix', 'celegans', 'nidulansM', 'TabacumC', 'cryptomonasC', 'musM', 'gallisepticum', 'syne', 'ecoli', 'subtilis', 'desulfuricans', 'reinhardtiiC', 'maritima', 'tenax', 'volcanii']

    
if loadmodel:
    model = keras.models.load_model(savename+'.hdf5', custom_objects = {
        'tf': tf,
        'weighted_binary_cross_entropy' : weighted_binary_cross_entropy})

else:
    inputs = Input(shape=(None, 5))

    h1 = Bidirectional(LSTM(50, return_sequences = True))(inputs)

    h1square = Lambda(SelfCartesian, output_shape = SelfCartesianShape)(h1)

    h2square_1 = Conv2D(filters=20, kernel_size=15, use_bias=False, padding='same')(h1square)
    h2square_2 = Conv2D(filters=20, kernel_size=9, use_bias=False, padding='same')(h1square)
    h2square_3 = Conv2D(filters=20, kernel_size=5, use_bias=False, padding='same')(h1square)
    h2square_a = concatenate([h2square_1, h2square_2, h2square_3], axis=-1)
    h2square_b = BatchNormalization(axis=-1)(h2square_a)
    h2square = Activation('relu')(h2square_b)

    h3square_1 = Conv2D(filters=20, kernel_size=9, use_bias=False, padding='same')(h2square)
    h3square_2 = Conv2D(filters=20, kernel_size=5, use_bias=False, padding='same')(h2square)
    h3square_a = concatenate([h3square_1, h3square_2], axis=-1)
    h3square_b = BatchNormalization(axis=-1)(h3square_a)
    h3square = Activation('relu')(h3square_b)

    h4square_1 = Conv2D(filters=20, kernel_size=5, activation='relu', padding='same')(h3square)
    sequencesquare = Lambda(SelfCartesian, output_shape = SelfCartesianShape)(inputs)
    h4square = concatenate([h4square_1, sequencesquare], axis = -1)

    output = Conv2D(filters=1, kernel_size=3, activation='sigmoid', padding='same')(h4square)

    opt = Adam(lr=0.0001)
    model = Model(input = inputs, output = output)
    model.compile(optimizer=opt,
                loss = weighted_binary_cross_entropy)

print(model.summary())


#batch_generator = batch_sub_generator(datafile, batchsize, length)
valid_x, valid_y = makebatch_sub(datafile, batchsize, length)
test_x, test_y = makebatch_sub('data/testset.txt', 3, None)


losses = []
validlosses = []
testlosses = []
# training loop
t = time.time()
for i in range(100):
    
    loss = model.fit_generator(batch_sub_generator_fit(datafile, batchsize, length), steps_per_epoch = 50)
    #losses.append(loss)
    
    validloss = model.evaluate(valid_x, valid_y, verbose = 0)
    validlosses.append(validloss)
    
    valid_yhat = np.squeeze(model.predict_on_batch(valid_x))
    valid_preds = np.rint(valid_yhat)
    
    plotresults(valid_y[0,:,:,0], outputdir+'valid-truth.png')
    plotresults(valid_yhat[0], outputdir+'valid-prob.png')
    plotresults(valid_preds[0], outputdir+'valid-pred.png')
    plotlosses(validlosses, validlosses = None, testlosses = None, name = outputdir+'losses.png')
    printoutputs(valid_y, valid_preds, i, validloss, theend = '\n\n')
    
    if i % 10 == 9:
        print('testoutputs')
        for j in range(16):
            test_x, test_y = makebatch_sub('data/testset.txt', 1, None, batchindices = [j])
            test_y = np.squeeze(test_y)
            test_yhat = np.squeeze(model.predict_on_batch(test_x))
            test_preds = np.rint(test_yhat)
            
            prefix = outputdir+'/testset/%02d_%s/%05d' % (j, testsetnames[j], i*50)
            
            if not os.path.exists(prefix[:-3]):
                os.makedirs(prefix[:-3])
            plotresults(test_y, prefix+'_truth.png')
            plotresults(test_yhat, prefix+'_prob.png')
            plotresults(test_preds, prefix+'_pred.png')
            
            printtestoutputs(test_y, test_preds, i*50, testsetnames[j], theend = '  ')
            print('    ppv:  %0.4f     sen:  %0.4f     acc:  %0.4f  ' % getaccuracy(test_y, test_yhat))
            
        print('\n')
    
        model.save(savename+'.hdf5')
        