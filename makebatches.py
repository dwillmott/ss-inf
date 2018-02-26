import numpy as np
import keras
from keras.utils import to_categorical


def getsamples(f, numbers):
    
    numbers = [n*5 for n in numbers]
    data = []
    
    for i, line in enumerate(f):
        if i-1 in numbers:
            sequence = line.rstrip().split(' ')
            sample = [sequence]
        if i-2 in numbers:
            structure = line.rstrip().split(' ')
            sample.append(structure)
        if i-3 in numbers:
            state = line.rstrip().split(' ')
            sample.append(state)
            data.append(sample)
    
    return data
    

def findsize(datafile):
    
    f = open(datafile, 'r')
    for i, line in enumerate(f):
        pass
    f.close()
    
    return int((i+1)/5)


def makebatch(datafile, batchsize, batchindices = None, totalsize = None, maxlength = None):
    # returns the tuple (batch_x, batch_y)
    
    if batchindices is None:
        if totalsize == None:
            totalsize = findsize(datafile)
        batchindices = np.random.choice(totalsize, batchsize, replace=False)
    
    f = open(datafile, 'r')
    
    data = getsamples(f, batchindices)
    
    lengths = [len(sample[0]) for sample in data]
    if maxlength == None:
        maxlength = max(lengths)
    
    # make x
    sequences = [sample[0] + (maxlength - length)*[5] for sample, length in zip(data, lengths)]
    sequencearray = np.stack([keras.utils.to_categorical(seq, num_classes=6) for seq in sequences])
    
    #make y
    z = []
    for sample in data:
        structure = sample[1]
        structurearray = np.zeros([len(structure), len(structure)])
        for i, j in enumerate(structure):
            if int(j):
                structurearray[i-1, int(j)-1] = 1
        
        structurearray = np.stack([1 - structurearray, structurearray], axis = -1)
        structurearray = np.pad(structurearray, [(0, maxlength - len(structure)), (0, maxlength - len(structure)), (0, 0)], 'constant')
        z.append(structurearray)
    
    z = np.stack(z)
    
    f.close()
    
    return sequencearray, z


def batch_generator(datafile, batchsize, length):
    totalsize = findsize(datafile)
    print(totalsize)
    indexlist = np.random.permutation(totalsize)
        
    while True:
        for i in range(0, totalsize//batchsize, batchsize):
            indices = indexlist[i:i+batchsize]
            yield makebatch(datafile, batchsize, indices, maxlength = length)

