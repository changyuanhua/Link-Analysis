import numpy as np
import time
from matplotlib import pyplot as plt

def read_file(filename):
    p = []
    p_ = []
    for line in open(filename, 'r'):
        x = line.split(",", 1)
        y = x[1].split("\n", 1)
        p.append(x[0])
        p_.append(y[0])
    for i in range(len(p)):
        p[i]=int(p[i])
        p_[i]=int(p_[i])
    z, z_ = np.array(p), np.array(p_)
    countz, countz_ = np.max(z), np.max(z_)
    if countz > countz_:
        count = countz
    else:
        count = countz_
    return p, p_, count

def read_outfile(filename):
    p = []
    p_ = []
    for line in open(filename, 'r'):
        x = line.split()
        p.append(x[1])
        p_.append(x[2])
    for i in range(len(p)):
        p[i]=int(p[i])
        p_[i]=int(p_[i])
    z, z_ = np.array(p), np.array(p_)
    countz, countz_ = np.max(z), np.max(z_)
    if countz > countz_:
        count = countz
    else:
        count = countz_
    return p, p_, count

def read_outfile_bi(filename):
    p = []
    p_ = []
    for line in open(filename, 'r'):
        x = line.split()
        p.append(x[1])
        p_.append(x[2])
    for i in range(len(p)):
        p[i]=int(p[i])
        p_[i]=int(p_[i])
    for i in range(len(p)):
        p.append(p_[i])
        p_.append(p[i])
    z, z_ = np.array(p), np.array(p_)
    countz, countz_ = np.max(z), np.max(z_)
    if countz > countz_:
        count = countz
    else:
        count = countz_
    return p, p_, count

def create_matrix(p, p_, count):
    a = [np.zeros(count) for i in range(count)]
    a = np.array(a)
    for i in range(len(p)):
        a[p[i]-1][p_[i]-1] = 1
    h = np.ones(count)
    return a, h

def hit(a, h):
    a_ = np.dot(a.T,h)
    h_ = np.dot(a,h)
    a_ = a_/np.sum(a_)
    h_ = h_/np.sum(h_)
    e = 1
    ite = 0
    while (e > 0.00001):
        past_a = a_
        past_h = h_
        a_ = np.dot(a.T,past_h)
        h_ = np.dot(a,past_a)
        a_ = a_/np.sum(a_)
        h_ = h_/np.sum(h_)
        e = np.linalg.norm(a_-past_a)+np.linalg.norm(h_-past_h)
        ite = ite + 1
    return a_, h_, ite

def output(a_, h_, count, tStart, tEnd, ite):
    num = np.zeros(count, dtype=int)
    for i in range(len(num)):
        num[i] = int(i+1)
    a_sort = np.lexsort([num,-a_])
    h_sort = np.lexsort([num,-h_])
    a_sort = [[num[i],a_[i]] for i in a_sort]
    h_sort = [[num[i],h_[i]] for i in h_sort]
    print('Authority','             ','Hub')
    for i in range(count):
        print('{:04d}'.format(a_sort[i][0]),': ','{:08e}'.format(a_sort[i][1]),'  ','{:04d}'.format(h_sort[i][0]),': ','{:08e}'.format(h_sort[i][1]))
    plt.bar(num-0.1, a_, width = 0.2, alpha =0.3, label='Authority')
    plt.bar(num+0.1, h_, width = 0.2, alpha =0.3, label='Hub')
    plt.legend(loc='upper right')
    plt.show()
    print('count:',count,'   executing hit function time:','{:08e}'.format( - tStart + tEnd ),'   hit function iteration counts:',ite)
    
if __name__ == "__main__":
    filename = 'hw3dataset/graph_1.txt'
    filename2 = 'hw3dataset/data.ntrans_0.1.nitems_0.1.1'
    #p, p_, count = read_file(filename)
    #p, p_, count = read_outfile(filename2)
    p, p_, count = read_outfile_bi(filename2)
    a, h = create_matrix(p, p_, count)
    tStart = time.time()
    a_, h_, ite = hit(a, h)
    tEnd = time.time()
    output(a_, h_, count, tStart, tEnd, ite)
