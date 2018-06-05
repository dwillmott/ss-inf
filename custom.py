import keras
import numpy as np
import time
from matplotlib import colors
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf
from sklearn.metrics import confusion_matrix

def SelfCartesian(x):
    newshape = K.stack([1, 1, K.shape(x)[1], 1])
    
    x_expanded = K.expand_dims(x, axis = -2)
    x_tiled = K.tile(x_expanded, newshape)
    x_transposed = K.permute_dimensions(x_tiled, (0,2,1,3))
    x_concat = K.concatenate([x_tiled, x_transposed], axis=-1)
    return x_concat


def SelfCartesianShape(input_shape):
    shape = list(input_shape)
    return [shape[0], shape[1], shape[1], shape[2]*2]


def weighted_binary_cross_entropy(labels, logits):
    weight = K.constant(200.0)
    class_weights = labels*weight + (1 - labels)
    unweighted_losses = K.binary_crossentropy(target=labels, output=logits)
    weighted_losses = unweighted_losses * class_weights
    
    loss = K.mean(tf.matrix_band_part(K.squeeze(weighted_losses, -1), 0, -1))
    return loss


def plotresults(square, name):
    norm = colors.Normalize(vmin=0., vmax=1.)
    
    fig = plt.figure()
    plt.imshow(np.triu(1 - square), norm = norm, interpolation='nearest')
    fig.savefig(name, dpi=300)
    plt.close(fig)
    return


def plotlosses(losses, validlosses, testlosses, name):
    fig = plt.figure()
    plt.plot(losses)
    if validlosses:
        plt.plot(validlosses)
    if testlosses:
        plt.plot(testlosses)
    fig.savefig(name)
    plt.close(fig)
    return

def printtestoutputs(y, pred, i, testname, theend):
    
    tn, fp, fn, tp = confusion_matrix(y[np.triu_indices(y.shape[1])].flatten(),
                                       pred[np.triu_indices(pred.shape[1])].flatten(),
                                       labels=[0,1]).ravel()
    
    #tn, fp, fn, tp = np.sum(confs, axis=0)
    print('{:4d}  {:20s}  '.format(i, testname), end='')
    print('   tn: {:7d}   fp: {:7d}   fn: {:3d}   tp: {:3d}'.format(tn, fp, fn, tp), end=theend)
    
    return

def printoutputs(batch_y, batch_preds, i, loss, theend):
    
    confs = np.stack([confusion_matrix(y[np.triu_indices(y.shape[1])].flatten(),
                                       pred[np.triu_indices(pred.shape[1])].flatten(),
                                       labels=[0,1]).ravel() for y, pred in zip(batch_y, batch_preds)])
    
    tn, fp, fn, tp = np.sum(confs, axis=0)
    print('{:4d}  {:5.5f}  '.format(i, loss), end='')
    print('   tn: {:7d}   fp: {:7d}   fn: {:4d}   tp: {:4d}'.format(tn, fp, fn, tp), end=theend)
    
    return

#def plotresults(batch_x, batch_y, batch_yhat, i):
    
    #seqlengths = np.argmin(np.sum(batch_x, axis=2), axis=1)
    #seqlengths = [seqlength if seqlength > 1 else batch_x.shape[1] for seqlength in seqlengths]
    #norm = colors.Normalize(vmin=0., vmax=1.)
    
    #fig, axes = plt.subplots(4,3)
    #for k, seqlength in enumerate(seqlengths[:4]):
        
        #axes[k,0].imshow(batch_y[k,:seqlength,:seqlength,0], norm = norm, interpolation='nearest')
        #axes[k,1].imshow(np.triu(batch_yhat[k,:seqlength,:seqlength,0]), norm = norm, interpolation='nearest')
        #axes[k,2].imshow(np.triu(batch_yhat[k,:seqlength,:seqlength,0]>0.5), norm = norm, interpolation='nearest')
    #fig.savefig("pictureoutput/ss-%05d.png" % (i,), dpi=200)
    #plt.close(fig)
    
    #return


def getpairs_rows(structure):
    structurethresh = (structure>0.5).astype(float)
    rowmax = np.argmax(structure, axis = 0)
    rowsum = np.sum(structurethresh, axis = 0)
    pairset = set([(j, r) for j, r in enumerate(rowmax) if rowsum[j]])
    return pairset


def getpairs_cols(structure):
    structurethresh = (structure>0.5).astype(float)
    rowmax = np.argmax(structure, axis = 1)
    rowsum = np.sum(structurethresh, axis = 1)
    pairset = set([(r, j) for j, r in enumerate(rowmax) if rowsum[j]])
    return pairset


def getaccuracy(y, yhat):
    
    seqlength = y.shape[1]
    
    truepairs = getpairs_rows(y)
    predictedpairs_rows = getpairs_rows(np.triu(yhat))
    
    rowmetrics = getmetrics(truepairs, predictedpairs_rows)
    
    truepairs = getpairs_cols(y)
    predictedpairs_cols = getpairs_cols(np.triu(yhat))
    colmetrics = getmetrics(truepairs, predictedpairs_cols)
        #print('     PPV: %0.3f     sen: %0.3f     acc: %0.3f      PPV: %0.3f     sen: %0.3f     acc: %0.3f' % (rowmetrics+colmetrics))
    
    return colmetrics# + colmetrics

#def getaccuracy(batch_x, batch_y, batch_yhat):
    
    #seqlengths = np.argmin(np.sum(batch_x, axis=2), axis=1)
    #seqlengths = [seqlength if seqlength > 1 else batch_x.shape[1] for seqlength in seqlengths]
    #for i in range(batch_y.shape[0]):
        #if seqlengths[i] == 0:
            #print(batch_x[i])
        #truepairs = getpairs_rows(batch_y[i, :seqlengths[i], :seqlengths[i], 1])
        
        #predictedpairs_rows = getpairs_rows(np.triu(batch_yhat[i, :seqlengths[i], :seqlengths[i], 1]))
        #rowmetrics = getmetrics(truepairs, predictedpairs_rows)
        #predictedpairs_cols = getpairs_rows(np.triu(batch_yhat[i, :seqlengths[i], :seqlengths[i], 1]))
        #colmetrics = getmetrics(truepairs, predictedpairs_cols)
        ##print('     PPV: %0.3f     sen: %0.3f     acc: %0.3f      PPV: %0.3f     sen: %0.3f     acc: %0.3f' % (rowmetrics+colmetrics))
    
    #return rowmetrics + colmetrics


def getmetrics(native, predicted, name = None):
    
    if not len(predicted):
        return 0.0, 0.0, 0.0
    tp = native.intersection(predicted)
    fn = native.difference(predicted)
    fp = predicted.difference(native)
    
    
    
    PPV = len(tp)/float(len(predicted))
    sen = len(tp)/float(len(native))
    accuracy = 0.5*(PPV + sen)
    
    return PPV, sen, accuracy


def checkpair(i, j):
    possiblepairs = [set([0, 3]),
                     set([1, 2]),
                     set([2, 3])]
                     
    return set([i,j]) in possiblepairs


def possiblepairs(batch_x):
    seqlengths = np.argmin(np.sum(batch_x, axis=2), axis=1)
    seqlengths = [seqlength if seqlength > 1 else batch_x.shape[1] for seqlength in seqlengths]
    
    z = []
    
    for i, x in enumerate(batch_x):
        x = x[:seqlengths[i]]
        
        xargs = np.argmax(x, axis = 1)
        
        structurearray = np.zeros([batch_x.shape[1], batch_x.shape[1]])
        
        for ind_i, ntide_i in enumerate(xargs):
            for ind_j, ntide_j in enumerate(xargs):
                if checkpair(ntide_i, ntide_j):
                    structurearray[ind_i,ind_j] = 1
        
        z.append(np.triu(structurearray))
    
    return np.stack(z)
