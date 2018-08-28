
import argparse
import numpy as np
import time, sys, os, datetime
from sklearn.metrics import confusion_matrix
from matplotlib import colors
import matplotlib.pyplot as plt


import tensorflow as tf
import keras
import keras.backend as k
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Lambda, Conv1D, Conv2D, Conv2DTranspose, Activation, Bidirectional, Concatenate, BatchNormalization, TimeDistributed
from keras.optimizers import RMSprop, Adam
from keras.regularizers import l2

from makebatches import *
from custom import *
from arch import *
from datanames import *

np.random.seed(32189)


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default = None, type=str, required = True)
parser.add_argument("--iterations", default = 20000, type = int)
parser.add_argument("--displaysteps", default = 5000, type = int)
parser.add_argument("--batchsize", default = 10, type = int)
parser.add_argument("--maxlength", default = 500, type = int) # None = no max length
parser.add_argument('--noBN', dest='BN', default=True, action='store_false')
parser.add_argument('--LSTMlayers', default = 1, type = int)
parser.add_argument("--weight", default = 5, type = int)
parser.add_argument("--reg", default = 0.00001, type = float)
parser.add_argument("--regtype", default = 'l2', type = str)
parser.add_argument("--lr", default= 0.001, type=float)
parser.add_argument("--load", default=False, action='store_true')
parser.add_argument("--threshold", default=0.5, type=float)
parser.add_argument("--lrdecay", default=True, action='store_false')
args = parser.parse_args()


dataset = args.dataset
iterations = args.iterations
lr = args.lr
reg = args.reg
loadmodel = args.load
maxlength = args.maxlength
batchsize = args.batchsize
weightint = args.weight
BN = args.BN
threshold = args.threshold
LSTMlayers = args.LSTMlayers
lrdecay = args.lrdecay


idstring = 'lr={:.0e}_reg={:.0e}_{:s}BN_LSTMlayers={:d}_weight={:d}_length={:d}'.format(lr, 
                                                                    reg, 
                                                                    'no'*(not BN),
                                                                    LSTMlayers, 
                                                                    weightint,
                                                                    maxlength)
today = datetime.datetime.today()
outputtopdir = 'outputs/{:s}_{:02d}_{:02d}'.format(dataset, today.month, today.day)
outputdir = outputtopdir+'/'+idstring+'/'
savename = 'saved/'+dataset+'/'+idstring

pathdict = {'strand' : ('data/strand/rnatrain.txt', 'data/strand/rnavalid.txt'),
            'strand16s' : ('data/strand/16s-finaltrain.txt', 'data/strand/16s-finalvalid.txt')}
trainpath, validpath = pathdict[dataset]
testpath = 'data/testdata/testdata.txt'
zspath = 'data/testdata/testset.txt'
writepath_train = outputdir+'trainlosses_'+idstring+'.txt'
writepath_valid = outputdir+'validlosses_'+idstring+'.txt'
writepath_test = outputdir+'testlosses_'+idstring+'.txt'

trainsize = findsize(trainpath)
validsize = findsize(validpath)
monitor_indices = np.random.choice(trainsize, trainsize//20, replace=False)

for path in ['outputs', outputtopdir, outputdir, 'saved', 'saved/'+dataset]:
    if not os.path.exists(path):
        os.makedirs(path)

#plt.gray()
zsnames, zsmfe, testsets, testsetnames, mfeaccuracy = getdatanames(dataset)
print(idstring+'   ', testsets)


if loadmodel:
    loadname = 'saved/strand16s/lr=1e-03_reg=1e-05_BN_LSTMlayers=1_weight=5_length=500_iter-20000.hdf5'
    model = keras.models.load_model(loadname, custom_objects = {
            'tf': tf,
            'weighted_binary_cross_entropy' : weighted_binary_cross_entropy})
    
    # get training, validation accuracy
    #trainmetrics = testonset(model, trainpath, writepath_train, monitor_indices, 'training set')
    #validmetrics = testonset(model, validpath, writepath_valid, range(validsize), 'validation set')
    
    totalstep = 20000
    
    # test sets
    #testfile = open(writepath_test, 'a+')
    #testfile.write('\n-----\ntest losses, iter {0:d}\n\n'.format(totalstep))
    #testfile.close()
    
    ## david set
    #davidsetmetrics = []
    #for k, (testset, testnames, mfeacc) in enumerate(zip(testsets, testsetnames, mfeaccuracy)):
        #davidsetmetrics += testonset(model, testpath, writepath_test, range(k*5, (k+1)*5), testset, testnames, mfeaccs = mfeacc)
    #writeavgmetrics(writepath_test, 'david 16s test total', davidsetmetrics)
    
    # zs set
    for thr in [0.999]:
        z = testonset(model, zspath, writepath_test, indices = [0], testsetname = 'sz', testnames = ['cuniculi_' + str(thr)], mfeaccs = [zsmfe[0]], threshold = thr)
        print('{:.2f} {:.3f} {:.3f}'.format(thr, z[0][0], z[0][1]))
    # write total test set metrics
    #writeavgmetrics(writepath_test, 'total', davidsetmetrics + zsmetrics)

    quit()


model, opt = makemodel(LSTMlayers, BN, weightint, reg, lr)

print(model.summary())


sample_x, sample_y = makebatch(trainpath, batchsize, maxlength)
sample_losses = []

# training loop
SPE = 100
for i in range(iterations//SPE):
    batchgen = batch_generator(trainpath, batchsize, maxlength)
    model.fit_generator(batchgen, steps_per_epoch = SPE)
    
    totalstep = (i+1)*SPE
    
    # print & plot sample batch losses
    sample_loss = model.evaluate(sample_x, sample_y, verbose = 0)
    sample_losses.append(sample_loss)
    plotlosses(sample_losses, validlosses = None, testlosses = None, name = outputdir+'losses_'+idstring+'.png', stepsize = SPE)
    
    #valid_yhat = np.squeeze(model.predict_on_batch(valid_x))
    #valid_preds = np.rint(valid_yhat)
    #validfile = open(outputdir+'validlosses_'+idstring+'.txt', 'a+')
    #printoutputs(valid_y, valid_preds, totalstep, validloss, validfile)
    #validfile.close()
    
    
    if i % 25 == 24:
        # save model
        model.save(savename+'_iter-{:05d}'.format(totalstep)+'.hdf5')
        
        #decay lr
        if lrdecay:
            newlr = 0.5*K.get_value(model.optimizer.lr)
            K.set_value(model.optimizer.lr, newlr)
            print('new lr: {0:f}'.format(newlr))
        
        # test everything
        if i > 75:
            # get training, validation accuracy
            trainmetrics = testonset(model, trainpath, writepath_train, monitor_indices, 'training set')
            validmetrics = testonset(model, validpath, writepath_valid, range(validsize), 'validation set')
            
            
            # test sets
            testfile = open(writepath_test, 'a+')
            testfile.write('\n-----\ntest losses, iter {0:d}\n\n'.format(totalstep))
            testfile.close()
            
            # david set
            davidsetmetrics = []
            for k, (testset, testnames, mfeacc) in enumerate(zip(testsets, testsetnames, mfeaccuracy)):
                davidsetmetrics += testonset(model, testpath, writepath_test, range(k*5, (k+1)*5), testset, testnames, mfeaccs = mfeacc)
            writeavgmetrics(writepath_test, 'david 16s test total', davidsetmetrics)
            
            # zs set
            zsmetrics = testonset(model, zspath, writepath_test, range(16), 'zs', mfeaccs = zsmfe)
            
            # write total test set metrics
            writeavgmetrics(writepath_test, 'total', davidsetmetrics + zsmetrics)
            
