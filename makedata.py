import numpy as np
import glob2

sequencedict = {'A' : '0', 'C' : '1', 'G' : '2', 'U' : '3'}
for letter in 'BDEFHIJKLMNOPQRSTVWXYZ':
    sequencedict.update({letter : '4'})


def getsequenceandstructure(filename, headersize):
    data = np.loadtxt(filename, skiprows = headersize, dtype='str')
    #print(data.shape)
    sequence = data[:,1]
    sequence = [sequencedict[s.upper()] for s in sequence]
    structure = data[:,4]
    state = structure.astype(bool).astype(int).astype(str)
    
    #print(' '.join(sequence))
    #print(' '.join(structure))
    #print(' '.join(state))
    
    return sequence, structure, state

#paths = glob2.glob('data/crw5s/*.ct')

globstring = 'data/raw/zs/*.ct'
outfile = 'data/zs.txt'

paths = glob2.glob(globstring)
f = open(outfile, 'w')

headersize = 5 # MAKE SURE THIS IS CORRECT!

for path in paths:
    sequence, structure, state = getsequenceandstructure(path, headersize)
    f.write(' '.join(sequence) + ' \n')
    f.write(' '.join(structure) + ' \n')
    f.write(' '.join(state) + ' \n')
    f.write('\n')