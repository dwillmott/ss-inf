import time

# slide one sequence along another, return largest number of matches
def strmatches(a, b):
    if len(a) > len(b):
        a, b = b, a
    
    difflist1 = [sum(1 for s, t in zip(a, b[-i:]) if s == t) for i in range(1, len(b)+1)]
    difflist2 = [sum(1 for s, t in zip(a[i:], b) if s == t) for i in range(1, len(a))]
    
    return max(difflist1+difflist2)


# gives longest common substring; modified from
# https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Longest_common_substring#Python
def longest_common_substring(s1, s2):
   m = [[0] * (1 + len(s2)) for i in range(1 + len(s1))]
   longest, x_longest = 0, 0
   for x in range(1, 1 + len(s1)):
       for y in range(1, 1 + len(s2)):
           if s1[x - 1] == s2[y - 1]:
               m[x][y] = m[x - 1][y - 1] + 1
               if m[x][y] > longest:
                   longest = m[x][y]
                   x_longest = x
   return [longest, float(longest)/len(s1), float(longest)/len(s2), x_longest - longest]


# compare two sequences S and Z. remove S if:
# 1) S and Z have a common substring of length len(Z)*0.1, or
# 2) there exists an alignment of S and Z where they match in more than len(Z)*0.75 places
def filtersequences(sequences, testsequences):
    
    outsequences = []
    t = time.time()
    
    for i, seq in enumerate(sequences):
        if i % (len(sequences)/100) == 0:
            print('\nChecked %5d of %5d, %ds\n' % (i, len(sequences), time.time() - t))
        
        for testseq in testsequences:
            d = strmatches(testseq, seq)
            
            if d > (len(testseq)*0.75):
                outsequences.append(i)
                print('{:4d}  {:3d}  {:3d}  {:3d}  '.format(i, len(seq), len(testseq), d))# longest_common_substring(testseq, seq))
                break
            
            lcs = longest_common_substring(testseq, seq)
            if lcs[1] > 0.1:
                print(i, len(seq), len(testseq), longest_common_substring(testseq, seq))
            
                
    return outsequences

trainpath = 'data/crw5s-comparative.txt'
testpath = 'data/testset-5s.txt'

trainfile = open(trainpath, 'r')
testfile = open(testpath, 'r')

trainsequences = []
for i, line in enumerate(trainfile):
    if i % 5 == 1:
        trainsequences.append(line.rstrip().split(' '))

trainfile.close()
        
testsequences = []
for i, line in enumerate(testfile):
    if i % 5 == 1:
        testsequences.append(line.rstrip().split(' '))
        
testfile.close()

print(len(trainsequences), len(testsequences))

filteredindices = filtersequences(trainsequences, testsequences)

print(len(filteredindices))

outfile = open('data/crw5s-filtered-2.txt', 'w')

trainfile = open(trainpath, 'r')

for i in range(len(trainsequences)):
    for j in range(5):
            line = trainfile.readline()
            if i not in filteredindices:
                outfile.write(line)

trainfile.close()
outfile.close()
