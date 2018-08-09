import keras
import numpy as np
import time
from matplotlib import colors
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from makebatches import *


# CUSTOM KERAS LAYERS

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


# PLOTTING

def plotresults(square, name):
    norm = colors.Normalize(vmin=0., vmax=1.)
    
    fig = plt.figure()
    plt.imshow(np.triu(1 - square), norm = norm, interpolation='nearest')
    fig.savefig(name, dpi=300)
    plt.close(fig)
    return


def plotlosses(losses, validlosses, testlosses, name, stepsize = 1):
    xax = range(0,len(losses)*stepsize, stepsize)
    fig = plt.figure()
    plt.plot(xax, losses)
    if validlosses:
        plt.plot(xax, validlosses)
    if testlosses:
        plt.plot(xax, testlosses)
    plt.ylim(0, 0.02)
    fig.savefig(name)
    plt.close(fig)
    return


def plotall(test_y, test_yhat, test_pred, j, name, step):
    
    prefix = outputdir+'/testset/%02d_%s/%05d' % (j, name, step)
    if not os.path.exists(prefix[:-3]):
        os.makedirs(prefix[:-3])
        
    plotresults(test_y, prefix+'_truth.png')
    plotresults(test_yhat, prefix+'_prob.png')
    plotresults(test_preds, prefix+'_pred.png')
    
    return


# COMBINE ALL TEST STEPS

def testonset(writefile, testpath, testsetname, testnames, indices, model, threshold, mfeaccs = None):
    writefile.write('\n{:15s} test set\n\n'.format(testsetname))
    if mfeaccs is None:
        mfeaccs = [None]*len(indices)
    
    metrics = []
    for testname, ind, mfeacc in zip(testnames, indices, mfeaccs):
        metrics.append(test_on_sequence(writefile, testpath, str(testname), ind, model, threshold, mfeacc))
    
    writeavgmetrics(writefile, testsetname, metrics)
        
    return metrics


def test_on_sequence(writefile, testpath, testname, ind, model, threshold, mfeacc = None):
    test_x, test_y, test_yhat, test_pred = get_xy(testpath, ind, model)
    
    tn, fp, fn, tp = get_confusion(test_y, test_pred)
    ppv, sen, acc = get_metrics(test_y, test_yhat, threshold)
    
    writeoutputs(writefile, testname, tn, fp, fn, tp, ppv, sen, acc, mfeacc)
    
    return ppv, sen, acc


# TP/TN/FP/FN

def get_confusion(y, pred):
    # return tn, fp, fn, tp
    return confusion_matrix(y[np.triu_indices(y.shape[1])].flatten(),
                                       pred[np.triu_indices(pred.shape[1])].flatten(),
                                       labels=[0,1]).ravel()


# WRITING

def writeoutputs(writefile, testname, tn, fp, fn, tp, ppv, sen, acc, mfeacc = None):
    writefile.write('{:20s} '.format(testname))
    writefile.write('tn: {:7d}  fp: {:7d}  fn: {:3d}  tp: {:3d}'.format(tn, fp, fn, tp))
    writefile.write(' ppv: {:0.3f}  sen: {:0.3f} ||'.format(tp/(tp+fp), tp/(tp+fn)))
    writefile.write(' ppv: {:0.3f}  sen: {:0.3f}  acc: {:0.3f}'.format(ppv, sen, acc))
    
    if not mfeacc is None:
        writefile.write('  mfe acc: {:0.3f}{:s}'.format(mfeacc, '  ***'*(acc < mfeacc)))
    
    writefile.write('\n')
    return

def writeavgmetrics(writefile, setname, metricslist):
    avgppv, avgsen, avgacc = np.mean(metricslist, axis = 0)
    writefile.write('\n{:15s} avg    ppv:  {:.4f}     sen:  {:.4f}     acc:  {:.4f}\n\n'.format(setname, avgppv, avgsen, avgacc))
    return


# PPV, SEN, ACC FUNCTIONS

def get_metrics(y, yhat, threshold = 0.5):
    
    truepairs = makepairs(y, threshold)
    predpairs = makepairs(np.triu(yhat), threshold)
    metrics = getmetrics_frompairs(set(truepairs), set(predpairs))
    
    return metrics


def makepairs(originalstructure, threshold = 0.5, nested = False):
    structure = np.copy(originalstructure)
    pairs = []
    while np.any(structure > 0.5):
        newpair = np.unravel_index(np.argmax(structure), structure.shape)
        #pairs.add(newpair)
        pairs.append(newpair)
        
        if nested:
            structure[:newpair[0],newpair[0]:newpair[1]+1] = 0
            structure[newpair[0]:newpair[1]+1,newpair[1]:] = 0
        else:
            structure[newpair[0]] = 0
            structure[:,newpair[1]] = 0
    
    return pairs


def getmetrics_frompairs(native, predicted):
    
    if not len(predicted) or not len(native):
        return 0.0, 0.0, 0.0
    tp = native.intersection(predicted)
    fn = native.difference(predicted)
    fp = predicted.difference(native)
    
    PPV = len(tp)/float(len(predicted))
    sen = len(tp)/float(len(native))
    accuracy = 0.5*(PPV + sen)
    
    return PPV, sen, accuracy


def get_xy(filename, ind, model):
    x, y = makebatch(filename, 1, None, batchindices = [ind])
    y = np.squeeze(y)
    yhat = np.triu(np.squeeze(model.predict_on_batch(x)))
    pred = np.rint(yhat)
    
    return x, y, yhat, pred


#  ---- NOT CURRENTLY USED ----

# PRINT SAMPLE TRAINING BATCH METRICS

def printoutputs(batch_y, batch_preds, step, loss, validfile):
    
    uppertri = np.triu_indices(batch_y.shape[1])
    
    confs = np.stack([confusion_matrix(y[uppertri].flatten(),
                                       pred[uppertri].flatten(),
                                       labels=[0,1]).ravel() for y, pred in zip(batch_y, batch_preds)])
    
    tn, fp, fn, tp = np.sum(confs, axis=0)
    printstring = '{:5d}  {:5.5f}     tn: {:7d}   fp: {:7d}   fn: {:4d}   tp: {:4d}'.format(step, loss, tn, fp, fn, tp)
    print(printstring)
    validfile.write(printstring+'\n')
    
    return

# CANONICAL BASE PAIRS

def checkpair(i, j):
    possiblepairs = [set([0, 3]),
                     set([1, 2]),
                     set([2, 3])]
                     
    return set([i,j]) in possiblepairs

def possiblepairs_batch(batch_x):
    seqlengths = np.argmin(np.sum(batch_x, axis=2), axis=1)
    seqlengths = [seqlength if seqlength > 1 else batch_x.shape[1] for seqlength in seqlengths]
    
    z = []
    
    for i, x in enumerate(batch_x):
        z.append(possiblepairs(x[:seqlengths[i]]))
    
    return np.stack(z)


def possiblepairs(batch_x):
    xargs = np.argmax(x, axis = 1)
    
    structurearray = np.zeros([batch_x.shape[1], batch_x.shape[1]])
    
    for ind_i, ntide_i in enumerate(xargs):
        for ind_j, ntide_j in enumerate(xargs):
            if checkpair(ntide_i, ntide_j):
                structurearray[ind_i,ind_j] = 1
    
    return np.triu(structurearray)
