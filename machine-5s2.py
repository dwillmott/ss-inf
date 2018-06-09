import argparse
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


parser = argparse.ArgumentParser()
parser.add_argument("--iterations", default = 5000, type = int)
parser.add_argument("--displaysteps", default = 50, type = int)
parser.add_argument("--batchsize", default = 10, type = int)
parser.add_argument('--noBN', dest='BN', default=True, action='store_false')
parser.add_argument('--useLSTM', dest='useLSTM', default=False, action='store_true')
parser.add_argument("--length", default = 500, type = int) # 0 = largest permissible
parser.add_argument("--weight", default = 50, type = int)
parser.add_argument("--reg", default = 0.0001, type = float)
parser.add_argument("--regtype", default = 'l2', type = str)
parser.add_argument("--lr", default= 0.001, type=float)
parser.add_argument("--load", default=False, type = bool)
parser.add_argument("--loadfile", default= '', type=str)

args = parser.parse_args()

lr = args.lr
reg = args.reg
loadmodel = args.load
batchsize = args.batchsize
weightint = args.weight
BN = args.BN
useLSTM = args.useLSTM

weight = k.constant(weightint)
l2reg = l2(reg)
datafile = 'data/crw5s-comparative.txt'
idstring = 'lr={:.0e}_reg={:.0e}_{:s}BN_weight={:d}{:s}'.format(lr, reg, 'no'*(not BN), weightint, '_noLSTM'*(not useLSTM))
outputdir = 'outputs5s/'+idstring+'/'
savename = 'saved/'+idstring+'.hdf5'
print(idstring)

for path in ['outputs', 'saved', outputdir]:
    if not os.path.exists(path):
        os.makedirs(path)
plt.gray()
#quit()

def weighted_binary_cross_entropy(labels, logits):
    class_weights = labels*weight + (1 - labels)
    unweighted_losses = K.binary_crossentropy(target=labels, output=logits)
    weighted_losses = unweighted_losses * class_weights
    
    loss = K.mean(tf.matrix_band_part(K.squeeze(weighted_losses, -1), 0, -1))
    return loss

testsetnames = ['cuniculi', 'vnecatrix', 'celegans', 'nidulansM',
                'TabacumC', 'cryptomonasC', 'musM', 'gallisepticum',
                'syne', 'ecoli', 'subtilis', 'desulfuricans',
                'reinhardtiiC', 'maritima', 'tenax', 'volcanii']

    
if loadmodel:
    model = keras.models.load_model(savename+'.hdf5', custom_objects = {
            'tf': tf,
            'weighted_binary_cross_entropy' : weighted_binary_cross_entropy})

else:
    inputs = Input(shape=(None, 5))
    
    if useLSTM:
        h1 = Bidirectional(LSTM(50, return_sequences = True))(inputs)
        h1square = Lambda(SelfCartesian, output_shape = SelfCartesianShape)(h1)
    else:
        h1square = Lambda(SelfCartesian, output_shape = SelfCartesianShape)(inputs)

    h2square_1 = Conv2D(filters=20, kernel_size=15, use_bias=False, kernel_regularizer = l2reg, padding='same')(h1square)
    h2square_2 = Conv2D(filters=20, kernel_size=9, use_bias=False, kernel_regularizer = l2reg, padding='same')(h1square)
    h2square_3 = Conv2D(filters=20, kernel_size=5, use_bias=False, kernel_regularizer = l2reg, padding='same')(h1square)
    h2square_a = concatenate([h2square_1, h2square_2, h2square_3], axis=-1)
    if BN:
        h2square_b = BatchNormalization(axis=-1)(h2square_a)
        h2square = Activation('relu')(h2square_b)
    else:
        h2square = Activation('relu')(h2square_a)

    h3square_1 = Conv2D(filters=20, kernel_size=9, use_bias=False, kernel_regularizer = l2reg, padding='same')(h2square)
    h3square_2 = Conv2D(filters=20, kernel_size=5, use_bias=False, kernel_regularizer = l2reg, padding='same')(h2square)
    h3square_a = concatenate([h3square_1, h3square_2], axis=-1)
    if BN:
        h3square_b = BatchNormalization(axis=-1)(h3square_a)
        h3square = Activation('relu')(h3square_b)
    else:
        h3square = Activation('relu')(h3square_a)

    h4square_1 = Conv2D(filters=20, kernel_size=5, activation='relu', kernel_regularizer = l2reg, padding='same')(h3square)
    sequencesquare = Lambda(SelfCartesian, output_shape = SelfCartesianShape)(inputs)
    h4square = concatenate([h4square_1, sequencesquare], axis = -1)

    output = Conv2D(filters=1, kernel_size=3, activation='sigmoid', kernel_regularizer = l2reg, padding='same')(h4square)

    opt = Adam(lr=lr)
    model = Model(input = inputs, output = output)
    model.compile(optimizer=opt, loss = weighted_binary_cross_entropy)

print(model.summary())

valid_x, valid_y = makebatch_sub(datafile, batchsize, None)

losses = []
validlosses = []
testlosses = []
# training loop
SPE = 100
for i in range(100):
    
    loss = model.fit_generator(batch_sub_generator_fit(datafile, batchsize, None), steps_per_epoch = SPE)
    
    validloss = model.evaluate(valid_x, valid_y, verbose = 0)
    validlosses.append(validloss)
    
    valid_yhat = np.squeeze(model.predict_on_batch(valid_x))
    valid_preds = np.rint(valid_yhat)
    plotlosses(validlosses, validlosses = None, testlosses = None, name = outputdir+'losses_'+idstring+'.png', step = SPE)
    
    validfile = open(outputdir+'validlosses_'+idstring+'.txt', 'a+')
    printoutputs(valid_y, valid_preds, i, SPE, validloss, validfile)
    validfile.close()
    
    if i % 5 == 4:
        metricslist = []
        testfile = open(outputdir+'testlosses_'+idstring+'.txt', 'a+')
        testfile.write('\ntest losses, iter {:d}\n\n'.format((i+1)*SPE))
        for j in range(16):
            
            test_x, test_y = makebatch_sub('data/testset.txt', 1, None, batchindices = [j])
            test_y = np.squeeze(test_y)
            test_yhat = np.squeeze(model.predict_on_batch(test_x))
            test_preds = np.rint(test_yhat)
            
            prefix = outputdir+'/testset/%02d_%s/%05d' % (j, testsetnames[j], i*SPE)
            
            if not os.path.exists(prefix[:-3]):
                os.makedirs(prefix[:-3])
            plotresults(test_y, prefix+'_truth.png')
            plotresults(test_yhat, prefix+'_prob.png')
            plotresults(test_preds, prefix+'_pred.png')
            
            metricslist.append(printtestoutputs(test_y, test_yhat, test_preds, i*SPE, testsetnames[j], testfile))
            
        averagemetrics = np.sum(metricslist, axis = 0)
        testfile.write('\nmin    ppv:  {0[0]:.4f}     sen:  {0[1]:.4f}     acc:  {0[2]:.4f}\n'.format(tuple(np.amin(metricslist, axis = 0))))
        testfile.write('max    ppv:  {0[0]:.4f}     sen:  {0[1]:.4f}     acc:  {0[2]:.4f}\n'.format(tuple(np.amax(metricslist, axis = 0))))
        testfile.write('avg    ppv:  {0[0]:.4f}     sen:  {0[1]:.4f}     acc:  {0[2]:.4f}\n'.format(tuple(np.mean(metricslist, axis = 0))))
        testfile.write('med    ppv:  {0[0]:.4f}     sen:  {0[1]:.4f}     acc:  {0[2]:.4f}\n\n\n'.format(tuple(np.median(metricslist, axis = 0))))
        testfile.close()
    
        model.save(savename)
        