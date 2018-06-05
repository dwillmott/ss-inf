import numpy as np
import keras
from keras.utils import to_categorical



def getallsamples(path):
    f = open(path, 'r')
    
    sequences = []
    states = []
    
    for i, line in enumerate(f):
        
        if i % 5 == 1:
            sequences.append(line.rstrip().split(' '))
        if i % 5 == 3:
            states.append(line.rstrip().split(' '))
    
    return sequences, states


def getsamples(f, numbers):
    
    # f: filename
    # number: indices of samples
    
    numbers = [n*5 for n in numbers] # samples take up five lines each
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
    
    return data # returns list of samples, sample is [sequence, structure, state]
    

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
    sequences = [sample[0][:maxlength] + (maxlength - length)*[5] for sample, length in zip(data, lengths)]
    sequencearray = np.stack([keras.utils.to_categorical(seq, num_classes=6) for seq in sequences])[:,:,:5]
    
    #make y
    z = []
    for sample in data:
        structure = sample[1][:maxlength]
        structurearray = np.zeros([len(structure), len(structure)])
        for i, j in enumerate(structure):
            if int(j) and int(j) <= maxlength:
                structurearray[i-1, int(j)-1] = 1
        
        structurearray = np.stack([np.triu(1 - structurearray), np.triu(structurearray)], axis = -1)
        structurearray = np.pad(structurearray, [(0, maxlength - len(structure)), (0, maxlength - len(structure)), (0, 0)], 'constant')
        z.append(structurearray)
    
    z = np.stack(z)
    
    f.close()
    
    return sequencearray, z

def makebatch_ofone(datafile, batchsize, batchindices = None, totalsize = None):
    # returns the tuple (batch_x, batch_y)
    
    if batchindices is None:
        if totalsize == None:
            totalsize = findsize(datafile)
        batchindices = np.random.choice(totalsize, batchsize, replace=False)
    
    f = open(datafile, 'r')
    
    data = getsamples(f, batchindices)
    
    sequencearray = []
    z = []
    for sequence, structure, state in data:
        length = len(sequence)
        
        sequence = keras.utils.to_categorical(sequence, num_classes=5)
        sequencearray.append(sequence)
        
        structurearray = np.zeros([length, length])
        
        for i, j in enumerate(structure):
                if int(j) and int(j) > i:
                    structurearray[i, int(j)-1] = 1
        z.append(structurearray)
    
    sequencearray = np.stack(sequencearray)
    z = np.stack(z)
    f.close()
    
    return sequencearray, z


def makebatch_sub(datafile, batchsize, sublength, batchindices = None, totalsize = None):
    # returns the tuple (batch_x, batch_y)
    
    if batchindices is None:
        if totalsize == None:
            totalsize = findsize(datafile)
        batchindices = np.random.choice(totalsize, batchsize, replace=False)
    
    f = open(datafile, 'r')
    
    data = getsamples(f, batchindices)
    
    if not sublength:
        sublength = min([len(sequence) for sequence, structure, state in data]) - 1
    
    sequencearray = []
    z = []
    for sequence, structure, state in data:
        length = len(sequence)
        start = np.random.randint(0, length - sublength)
        subsequence = keras.utils.to_categorical(sequence[start:start+sublength], num_classes=5)
        sequencearray.append(subsequence)
        
        substructure = structure[start:start+sublength]
        substructurearray = np.zeros([sublength, sublength])
        
        for i, j in enumerate(structure):
            if (i > start) and (i < (start+sublength)):
                if int(j) and int(j) > i and int(j) <= (start+sublength):
                    substructurearray[i-start, int(j)-start-1] = 1
        z.append(substructurearray)
    
    sequencearray = np.stack(sequencearray)
    z = np.expand_dims(np.stack(z), -1)
    f.close()
    
    return sequencearray, z

def batch_sub_generator(datafile, batchsize, length):
    totalsize = findsize(datafile)
    totalsize = (totalsize//batchsize)*batchsize
    indexlist = np.random.permutation(totalsize)
        
    while True:
        for i in range(0, totalsize, batchsize):
            indices = indexlist[i:i+batchsize]
            yield makebatch_sub(datafile, batchsize, length, indices, totalsize)

def batch_sub_generator_fit(datafile, batchsize, length):
    totalsize = findsize(datafile)
    totalsize = (totalsize//batchsize)*batchsize
    indexlist = np.random.permutation(totalsize)
    
    for i in range(0, totalsize, batchsize):
        indices = indexlist[i:i+batchsize]
        yield makebatch_sub(datafile, batchsize, length, indices, totalsize)

def batch_generator(datafile, batchsize, length = None):
    totalsize = findsize(datafile)
    totalsize = (totalsize//batchsize)*batchsize
    indexlist = np.random.permutation(totalsize)
        
    while True:
        for i in range(0, totalsize, batchsize):
            indices = indexlist[i:i+batchsize]
            yield makebatch(datafile, batchsize, indices, maxlength = length)

