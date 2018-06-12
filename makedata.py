import numpy as np
import glob


def converttonum(letter):
    letter = letter.upper()
    if letter in 'ACGU':
        return str('ACGU'.index(letter))
    else:
        return '4'


def getsequenceandstructure(filename):
    print(filename)
    with open(filename, 'r') as f2:
        lines = (line for line in f2 if (line.lstrip()[0].isnumeric() and 'energy' not in line.lower() and '=' not in line))
    #f2.close()
        data = np.loadtxt(lines, dtype='str')
    
    sequence = data[:,1]
    sequence = [converttonum(s) for s in sequence]
    structure = data[:,4]
    state = structure.astype(bool).astype(int).astype(str)
    
    return sequence, structure, state


def writedatafile(paths, outfile):
    f = open(outfile, 'w')
    
    for path in paths:
        sequence, structure, state = getsequenceandstructure(path)
        if sum([int(st) for st in state]):
            f.write(path + '\n')
            f.write(' '.join(sequence) + ' \n')
            f.write(' '.join(structure) + ' \n')
            f.write(' '.join(state) + ' \n')
            f.write('\n')

    f.close()
    return

if __name__ == '__main__':
    
    # CHANGE THESE IF YOU'RE USING YOUR OWN DATA
    datadirectory = 'testdata'  # directory with .ct files
    outfile = 'testdata.txt'  # output file to write to
    headersize = 5  # number of lines in the .ct file before the sequence begins
    
    
    # get all filepaths
    ctglobstring = datadirectory + '/**/*.ct'
    nopctglobstring = datadirectory + '/**/*.nopct'
    paths = glob.glob(ctglobstring, recursive = True) + glob.glob(nopctglobstring, recursive = True)
    
    writedatafile(paths, outfile)
    