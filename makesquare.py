import numpy as np


np.set_printoptions(threshold=100000, linewidth=400)
A = np.load('data/zs.npy')

def getpairs(structfile):
    real_struct_file = open(structfile, 'r');

    real_seq = []; 
    real_pairing = [];
    pairs = []

    firstline = real_struct_file.readline();

    for line in real_struct_file.readlines():
        splitline = line.split();
        position = int(splitline[0])
        pairedwith = int(splitline[4])
        if pairedwith and position < pairedwith:
            pairs.append((position, pairedwith))
    
    return pairs, position

zsnames = ['cuniculi', 'vnecatrix', 'celegansM', 'nidulansM',
            'tabacumC', 'cryptomonasC', 'musM', 'gallisepticum',
            'syne', 'ecoli', 'subtilis', 'desulfuricans',
            'reinhardtiiC', 'maritima', 'tenax', 'volcanii']

z = []

for zs in zsnames:
    pairs, length = getpairs('data/raw/zs/%s-native-nop.ct' % (zs,))
    outarray = np.zeros([length, length, 3])
    print(outarray.shape)
    for i in range(length):
        for j in range(length):
            #print(i, j)
            outarray[i,j,0] = i+1 + (j+1)*10000
            #outarray[i,j,1] = j+1
    for p in pairs:
        print(p)
        print((p[1],p[0]))
        outarray[p + (1,)] = 1
        outarray[(p[1], p[0], 1)] = 1
        print(np.count_nonzero(outarray[:,:,1]))
    outarray[:,:,2] = 1 - outarray[:,:,1]
    print(length)
    
    outarray = np.pad(outarray, ((0, 1827 - length), (0, 1827 - length), (0,0)), 'constant')
    
    #print(outarray[:50, :50, 2])
    
    z.append(outarray)

z = np.stack(z)
print(z.shape)

print(z[5, 657:664, 657:664])

np.save('zs-square.npy', z)

#A = A[:,:,[0, -3, -2, -1]]
#print(A[4,:30])

#A = A[:,:,None,:]

#A = np.repeat(A, A.shape[1], axis = -2)

#print(A.shape)

#newA = np.concatenate([A, A.transpose((0,2,1,3))], axis = -1)
#print(newA.shape)

#print(newA[5, :8, :8])

#newA[:,:,:,2] = np.logical_and(newA[:,:,:,2], newA[:,:,:,6]).astype(int)

#print(newA[5, 8:14, 8:14])