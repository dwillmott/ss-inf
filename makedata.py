import numpy as np
import glob2

sequencedict = {'A' : '0', 'C' : '1', 'G' : '2', 'U' : '3'}
for letter in 'BDEFHIJKLMNOPQRSTVWXYZ':
    sequencedict.update({letter : '4'})


def getsequenceandstructure(filename, headersize):
    data = np.loadtxt(filename, skiprows = headersize, dtype='str')
    
    sequence = data[:,1]
    sequence = [sequencedict[s.upper()] for s in sequence]
    structure = data[:,4]
    state = structure.astype(bool).astype(int).astype(str)
    
    return sequence, structure, state

def writedatafile(globstring, outfile, headersize):
    paths = glob2.glob(globstring)
    
    f = open(outfile, 'w')
    
    for path in paths:
        sequence, structure, state = getsequenceandstructure(path, headersize)
        f.write(path + '\n')
        f.write(' '.join(sequence) + ' \n')
        f.write(' '.join(structure) + ' \n')
        f.write(' '.join(state) + ' \n')
        f.write('\n')

    f.close()
    return

if __name == '__main__':
    
    globstring = 'data/raw/crw5s-comparative/*/*.nopct'
    outfile = 'data/crw5s-comparative.txt'
    headersize = 5
    
    writedatafile(globstring, outfile, headersize)
    