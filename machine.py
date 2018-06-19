import argparse
import keras
import numpy as np
import time
import sys, os
from sklearn.metrics import confusion_matrix
from matplotlib import colors
import matplotlib.pyplot as plt
import keras.backend as k
import datetime

from keras.models import Model
from keras.layers import Input, Dense, LSTM, Lambda, Conv1D, Conv2D, Conv2DTranspose, Activation, Bidirectional, Concatenate, BatchNormalization, TimeDistributed
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
parser.add_argument("--maxlength", default = 500, type = int) # None = no max length
parser.add_argument('--noBN', dest='BN', default=True, action='store_false')
parser.add_argument('--LSTMlayers', default = 0, type = int)
parser.add_argument("--weight", default = 20, type = int)
parser.add_argument("--reg", default = 0.00001, type = float)
parser.add_argument("--regtype", default = 'l2', type = str)
parser.add_argument("--lr", default= 0.0001, type=float)
parser.add_argument("--load", default=False, type = bool)
parser.add_argument("--threshold", default=0.5, type=float)
parser.add_argument("--dataset", type=str)
args = parser.parse_args()

lr = args.lr
reg = args.reg
loadmodel = args.load
maxlength = args.maxlength
batchsize = args.batchsize
weightint = args.weight
BN = args.BN
threshold = args.threshold
LSTMlayers = args.LSTMlayers
dataset = args.dataset
testpath = 'testdata/testdata.txt'

dataset_dict = {'strand' : 'strand/strand-filtered.txt',
                'strand16s' : 'strand/16s-filtered.txt'}
datapath = dataset_dict[dataset]

weight = k.constant(weightint)
l2reg = l2(reg)

idstring = 'lr={:.0e}_reg={:.0e}_{:s}BN_LSTMlayers={:d}_weight={:d}'.format(lr, 
                                                                    reg, 
                                                                    'no'*(not BN),
                                                                    LSTMlayers, 
                                                                    weightint)
today = datetime.datetime.today()
outputtopdir = 'outputs_{:s}_{:02d}_{:02d}'.format(dataset, today.month, today.day)
outputdir = outputtopdir+'/'+idstring+'/'
savename = 'saved/'+dataset+'/'+idstring+'.hdf5'
print('\n'+idstring+'\n')

for path in [outputtopdir, outputdir, 'saved', 'saved/'+dataset]:
    if not os.path.exists(path):
        os.makedirs(path)
plt.gray()

def weighted_binary_cross_entropy(labels, logits):
    class_weights = labels*weight + (1 - labels)
    unweighted_losses = K.binary_crossentropy(target=labels, output=logits)
    
    weighted_losses = unweighted_losses * class_weights
    
    loss = K.mean(tf.matrix_band_part(K.squeeze(weighted_losses, -1), 0, -1))
    return loss

testsets = ['16s_small', '16s_extra', '16s_long', '16s_med', 'rnasep', 'intron', '5s']

testsetnames = ['V.ursinus', 'S.aestuans', 'L.catta', 'N.robinsoni', 'A.cahirinus',
                'P.vivax', 'R.carriebowensis', 'O.cuniculus', 'P.falciparum', 'Z.mays',
                'S.griseus', 'M.leprae', 'E.coli', 'C.testosteroni', 'M.hyopneumoniae',
                'V.acridophagus', 'V.corneae', 'E.schubergi', 'V.imperfecta', 'E.cuniculi',
                'H.chlorum', 'T.syrichta', 'P.fluorescens', 'Z.bailii', 'A.ferrooxidans',
                'H.rubra', 'S.anglica', 'B.yamatoana', 'T.thermophila', 'P.thunbergii',
                'S.pombe', 'P.waltl', 'O.sativa', 'M.glyptostroboides', 'M.fossilis']

mfeaccuracy = [0.135, 0.34, 0.251, 0.447, 0.20,
               0.385, 0.338, 0.177, 0.423, 0.258,
               0.322, 0.179, 0.41, 0.524, 0.639,
               0.371, 0.33, 0.23, 0.288, 0.17,
               0.32, 0.13, 0.49, 0.68, 0.59,
               0.30, 0.06, 0.51, 0.74, 0.13,
               0.85, 0.76, 0.55, 0.29, 0.15]

    
if loadmodel:
    model = keras.models.load_model(savename, custom_objects = {
            'tf': tf,
            'weighted_binary_cross_entropy' : weighted_binary_cross_entropy})
    
    testfile = open(outputdir+'testlosses_'+idstring+'.txt', 'a+')
    testfile.write('\n-----\ntest losses, end\n\n')
    testmetrics = []
        
    for k, testset in enumerate(testsets[:4]):
        
        testfile.write('\n{:15s} test set\n\n'.format(testset))
        
        for j in range(k*5, (k+1)*5):
            testmetrics.append(test_on_sequence(testfile, testpath, testname[j], j, model, threshold, mfeacc[j]))
        
        writeavgmetrics(testfile, testset, testmetrics[-5:])
    
    writeavgmetrics(testfile, 'total', testmetrics)
    testfile.close()

    quit()

else:
    inputs = Input(shape=(None, 5))
    
    if LSTMlayers:
        h1_lstm = Bidirectional(LSTM(75, return_sequences = True))(inputs)
        if LSTMlayers > 1:
            h1_lstm = Bidirectional(LSTM(50, return_sequences = True))(h1_lstm)
        h1_lstmout = TimeDistributed(Dense(20))(h1_lstm)
        h1 = Concatenate(axis=-1)([inputs, h1_lstmout])
        h1square = Lambda(SelfCartesian, output_shape = SelfCartesianShape)(h1)
    else:
        h1square = Lambda(SelfCartesian, output_shape = SelfCartesianShape)(inputs)

    h2square_1 = Conv2D(filters=20, kernel_size=15, use_bias=False, kernel_regularizer = l2reg, padding='same')(h1square)
    h2square_2 = Conv2D(filters=20, kernel_size=9, use_bias=False, kernel_regularizer = l2reg, padding='same')(h1square)
    h2square_3 = Conv2D(filters=20, kernel_size=5, use_bias=False, kernel_regularizer = l2reg, padding='same')(h1square)
    h2square_a = Concatenate(axis=-1)([h2square_1, h2square_2, h2square_3])
    if BN:
        h2square_b = BatchNormalization(axis=-1)(h2square_a)
        h2square = Activation('relu')(h2square_b)
    else:
        h2square = Activation('relu')(h2square_a)

    h3square_1 = Conv2D(filters=20, kernel_size=9, use_bias=False, kernel_regularizer = l2reg, padding='same')(h2square)
    h3square_2 = Conv2D(filters=20, kernel_size=5, use_bias=False, kernel_regularizer = l2reg, padding='same')(h2square)
    h3square_a = Concatenate(axis=-1)([h3square_1, h3square_2])
    if BN:
        h3square_b = BatchNormalization(axis=-1)(h3square_a)
        h3square = Activation('relu')(h3square_b)
    else:
        h3square = Activation('relu')(h3square_a)

    h4square_1 = Conv2D(filters=20, kernel_size=5, activation='relu', kernel_regularizer = l2reg, padding='same')(h3square)
    sequencesquare = Lambda(SelfCartesian, output_shape = SelfCartesianShape)(inputs)
    h4square = Concatenate(axis=-1)([h4square_1, sequencesquare])

    output = Conv2D(filters=1, kernel_size=3, activation='sigmoid', kernel_regularizer = l2reg, padding='same')(h4square)

    opt = Adam(lr=lr)
    model = Model(input = inputs, output = output)
    model.compile(optimizer=opt, loss = weighted_binary_cross_entropy)

print(model.summary())

valid_x, valid_y = makebatch(datapath, batchsize, maxlength)

validlosses = []
# training loop
SPE = 10
for i in range(200):
    model.fit_generator(batch_generator(datapath, batchsize, maxlength), steps_per_epoch = SPE)
    
    totalstep = (i+1)*SPE
    
    validloss = model.evaluate(valid_x, valid_y, verbose = 0)
    validlosses.append(validloss)
    
    valid_yhat = np.squeeze(model.predict_on_batch(valid_x))
    valid_preds = np.rint(valid_yhat)
    plotlosses(validlosses, validlosses = None, testlosses = None, name = outputdir+'losses_'+idstring+'.png', stepsize = SPE)
    
    validfile = open(outputdir+'validlosses_'+idstring+'.txt', 'a+')
    printoutputs(valid_y, valid_preds, totalstep, validloss, validfile)
    validfile.close()
    
    if i % 20 == 0:
        testfile = open(outputdir+'testlosses_'+idstring+'.txt', 'a+')
        testfile.write('\n-----\ntest losses, iter {:d}\n\n'.format(totalstep))
        testmetrics = []
        
        for k, testset in enumerate(testsets[:4]):
            
            testfile.write('\n{:15s} test set\n\n'.format(testset))
            
            for j in range(k*5, (k+1)*5):
                testmetrics.append(test_on_sequence(testfile, testpath, testsetnames[j], j, model, threshold, mfeaccuracy[j]))
            
            writeavgmetrics(testfile, testset, testmetrics[-5:])
        
        writeavgmetrics(testfile, 'total', testmetrics)
        testfile.close()
    
        model.save(savename)
        