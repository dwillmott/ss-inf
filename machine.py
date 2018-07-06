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

np.random.seed(32189)


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default = None, type=str, required = True)
parser.add_argument("--iterations", default = 20000, type = int)
parser.add_argument("--displaysteps", default = 5000, type = int)
parser.add_argument("--batchsize", default = 10, type = int)
parser.add_argument("--maxlength", default = 500, type = int) # None = no max length
parser.add_argument('--noBN', dest='BN', default=True, action='store_false')
parser.add_argument('--LSTMlayers', default = 0, type = int)
parser.add_argument("--weight", default = 1, type = int)
parser.add_argument("--reg", default = 0.00001, type = float)
parser.add_argument("--regtype", default = 'l2', type = str)
parser.add_argument("--lr", default= 0.0001, type=float)
parser.add_argument("--load", default=False, type = bool)
parser.add_argument("--threshold", default=0.5, type=float)
parser.add_argument("--lrdecay", default=False, action='store_true')
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
testpath = 'data/testdata/testdata.txt'
zspath = 'data/testdata/testset.txt'
randompath = 'data/testdata/16s-randomtest.txt'

dataset_dict = {'strand' : 'data/strand/strand-filtered.txt',
                'strand16s' : 'data/strand/16s-filtered.txt',
                'strand16s-both' : 'data/strand/16s-filtered_both.txt',
                'strand16s-random' : 'data/strand/16s-filtered_random.txt'}
datapath = dataset_dict[dataset]

idstring = 'lr={:.0e}_reg={:.0e}_{:s}BN_LSTMlayers={:d}_weight={:d}_length={:d}'.format(lr, 
                                                                    reg, 
                                                                    'no'*(not BN),
                                                                    LSTMlayers, 
                                                                    weightint,
                                                                    maxlength)
today = datetime.datetime.today()
outputtopdir = 'outputs/{:s}_{:02d}_{:02d}'.format(dataset, today.month, today.day)
outputdir = outputtopdir+'/'+idstring+'/'
savename = 'saved/'+dataset+'/'+idstring+'.hdf5'
print('\n'+idstring+'\n')
print(lrdecay)

for path in ['outputs', outputtopdir, outputdir, 'saved', 'saved/'+dataset]:
    if not os.path.exists(path):
        os.makedirs(path)
plt.gray()


zsnames = ['cuniculi', 'vnecatrix', 'celegans', 'nidulansM',
           'TabacumC', 'cryptomonasC', 'musM', 'gallisepticum',
           'syne', 'ecoli', 'subtilis', 'desulfuricans',
           'reinhardtiiC', 'maritima', 'tenax', 'volcanii']

zsmfe = [0.171, 0.181, 0.203, 0.272,
         0.323, 0.339, 0.375, 0.385,
         0.361, 0.411, 0.512, 0.533,
         0.537, 0.562, 0.618, 0.752]

testsets = ['16s_small', '16s_extra', '16s_long', '16s_med']

testsetnames = [['V.ursinus', 'S.aestuans', 'L.catta', 'N.robinsoni', 'A.cahirinus'],
                ['P.vivax', 'R.carriebowensis', 'O.cuniculus', 'P.falciparum', 'Z.mays'],
                ['S.griseus', 'M.leprae', 'E.coli', 'C.testosteroni', 'M.hyopneumoniae'],
                ['V.acridophagus', 'V.corneae', 'E.schubergi', 'V.imperfecta', 'E.cuniculi']]

mfeaccuracy = [[0.135, 0.34, 0.251, 0.447, 0.20],
               [0.385, 0.338, 0.177, 0.423, 0.258],
               [0.322, 0.179, 0.41, 0.524, 0.639],
               [0.371, 0.33, 0.23, 0.288, 0.17]]

if dataset == 'strand':
    testsets += ['rnasep', 'intron', '5s']

    testsetnames += [['H.chlorum', 'T.syrichta', 'P.fluorescens', 'Z.bailii', 'A.ferrooxidans'],
                    ['H.rubra', 'S.anglica', 'B.yamatoana', 'T.thermophila', 'P.thunbergii'],
                    ['S.pombe', 'P.waltl', 'O.sativa', 'M.glyptostroboides', 'M.fossilis']]

    mfeaccuracy += [[0.32, 0.13, 0.49, 0.68, 0.59],
                   [0.30, 0.06, 0.51, 0.74, 0.13],
                   [0.85, 0.76, 0.55, 0.29, 0.15]]

if loadmodel:
    testfile = open(outputdir+'testlosses_'+idstring+'.txt', 'a+')
    testfile.write('\n-----\ntest losses\n\n')
    
    if dataset == 'strand16s-both':
        # david set
        davidsetmetrics = []
        for k, (testset, testnames, mfeacc) in enumerate(zip(testsets, testsetnames, mfeaccuracy)):
            subsetmetrics = testonset(testfile, testpath, testset, testnames, range(k*5, (k+1)*5), model, threshold, mfeacc)
            davidsetmetrics += subsetmetrics
        writeavgmetrics(testfile, 'david 16s test total', davidsetmetrics)
        
        # zs set
        zsmetrics = testonset(testfile, zspath, 'zs', zsnames, range(16), model, threshold, zsmfe)
        
        # write total metrics
        writeavgmetrics(testfile, 'total', davidsetmetrics + zsmetrics)
    
    # random set
    if dataset == 'strand16s-random':
        testonset(testfile, randompath, 'random', range(1, 50), range(49), model, threshold, mfeaccs = None)
    
    testfile.close()

    quit()


model, opt = makemodel(LSTMlayers, BN, weightint, reg, lr)

print(model.summary())

trainsize = findsize(datapath)
monitor_indices = np.random.choice(trainsize, trainsize//10, replace=False)

valid_x, valid_y = makebatch(datapath, batchsize, maxlength)
validlosses = []

# training loop
SPE = 100
for i in range(iterations//SPE):
    print(K.get_value(model.optimizer.lr))
    model.fit_generator(batch_generator(datapath, batchsize, maxlength), steps_per_epoch = SPE)
    
    totalstep = (i+1)*SPE
    
    # print & plot validation losses
    validloss = model.evaluate(valid_x, valid_y, verbose = 0)
    validlosses.append(validloss)
    
    valid_yhat = np.squeeze(model.predict_on_batch(valid_x))
    valid_preds = np.rint(valid_yhat)
    plotlosses(validlosses, validlosses = None, testlosses = None, name = outputdir+'losses_'+idstring+'.png', stepsize = SPE)
    
    validfile = open(outputdir+'validlosses_'+idstring+'.txt', 'a+')
    printoutputs(valid_y, valid_preds, totalstep, validloss, validfile)
    validfile.close()
    
    if lrdecay:
        if i % 25 == 24:
            newlr = 0.5*K.get_value(model.optimizer.lr)
            K.set_value(model.optimizer.lr, newlr)
    
    if i > 75 and i % 25 == 24:
        validfile = open(outputdir+'validlosses_'+idstring+'.txt', 'a+')
        testonset(validfile, datapath, 'train set', range(1, (trainsize//10)+1), range(trainsize//10), model, threshold)
        validfile.close()
        
        testfile = open(outputdir+'testlosses_'+idstring+'.txt', 'a+')
        testfile.write('\n-----\ntest losses, iter {:d}\n\n'.format(totalstep))
        
        if dataset == 'strand16s-both':
            # david set
            davidsetmetrics = []
            for k, (testset, testnames, mfeacc) in enumerate(zip(testsets, testsetnames, mfeaccuracy)):
                davidsetmetrics += testonset(testfile, testpath, testset, testnames, range(k*5, (k+1)*5), model, threshold, mfeacc)
            writeavgmetrics(testfile, 'david 16s test total', davidsetmetrics)
            
            # zs set
            zsmetrics = testonset(testfile, zspath, 'zs', zsnames, range(16), model, threshold, zsmfe)
            
            # write total metrics
            writeavgmetrics(testfile, 'total', davidsetmetrics + zsmetrics)
        
        # random set
        if dataset == 'strand16s-random':
            testonset(testfile, randompath, 'random', range(1, 50), range(49), model, threshold, mfeaccs = None)
        
        testfile.close()
    
        model.save(savename)
