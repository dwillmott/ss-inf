#import keras
import numpy as np
#import time
from matplotlib import colors
import matplotlib.pyplot as plt


def plotresults(batch_x, batch_y, batch_yhat, i):
    
    seqlengths = np.argmin(np.sum(batch_x, axis=2), axis=1)
    seqlengths = [seqlength if seqlength > 1 else batch_x.shape[1] for seqlength in seqlengths]
    norm = colors.Normalize(vmin=0., vmax=1.)
    
    fig, axes = plt.subplots(4,3)
    for k, seqlength in enumerate(seqlengths[:4]):
        
        axes[k,0].imshow(batch_y[k,:seqlength,:seqlength,0], norm = norm, interpolation='nearest')
        axes[k,1].imshow(np.triu(batch_yhat[k,:seqlength,:seqlength,0]), norm = norm, interpolation='nearest')
        axes[k,2].imshow(np.triu(batch_yhat[k,:seqlength,:seqlength,0]>0.5), norm = norm, interpolation='nearest')
    fig.savefig("pictureoutput/ss-%05d.png" % (i,), dpi=200)
    plt.close(fig)
    
    return


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


def getaccuracy(batch_x, batch_y, batch_yhat):
    
    seqlengths = np.argmin(np.sum(batch_x, axis=2), axis=1)
    for i in range(batch_y.shape[0]):
        if seqlengths[i] == 0:
            print(batch_x[i])
        truepairs = getpairs_rows(batch_y[i, :seqlengths[i], :seqlengths[i], 1])
        
        predictedpairs_rows = getpairs_rows(np.triu(batch_yhat[i, :seqlengths[i], :seqlengths[i], 1]))
        rowmetrics = getmetrics(truepairs, predictedpairs_rows)
        predictedpairs_cols = getpairs_rows(np.triu(batch_yhat[i, :seqlengths[i], :seqlengths[i], 1]))
        colmetrics = getmetrics(truepairs, predictedpairs_cols)
        #print('     PPV: %0.3f     sen: %0.3f     acc: %0.3f      PPV: %0.3f     sen: %0.3f     acc: %0.3f' % (rowmetrics+colmetrics))
    
    return rowmetrics + colmetrics


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
