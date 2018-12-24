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
    ee = [np.ones(count,dtype=int) for i in range(count)]
    a = np.array(a)
    ee = np.array(ee)
    for i in range(len(p)):
        a[p[i]-1][p_[i]-1] = 1
    for i in range(count):
        if (np.sum(a[i]) != 0):
            a[i] = a[i] / np.sum(a[i])
    pt = a.T
    ee = ee/count
    a = 0.85 * pt + 0.15 * ee
    x = np.ones(count)
    return a, x

def pagerank(a, x):
    r = np.dot(a,x)
    r = r / np.sum(r)
    e = 1
    ite = 0
    while (e > 0.00001):
        past_r = r
        r = np.dot(a,r)
        r = r / np.sum(r)
        e = np.linalg.norm(r-past_r)
        ite = ite + 1
    return r, ite

def output(r, count, tStart, tEnd, ite):
    num = np.zeros(count, dtype=int)
    for i in range(len(num)):
        num[i] = int(i+1)
    r_sort = np.lexsort([num,-r])
    r_sort = [[num[i],r[i]] for i in r_sort]
    print('Page rank')
    for i in range(count):
        print('{:04d}'.format(r_sort[i][0]),': ','{:08e}'.format(r_sort[i][1]))
    plt.bar(num, r, width = 0.6, alpha =0.3)
    plt.show()
    print('count:',count,'   executing hit function time:','{:08e}'.format( - tStart + tEnd ),'   hit function iteration counts:',ite)
    
if __name__ == "__main__":
    filename = 'hw3dataset/graph_add3.txt'
    filename2 = 'hw3dataset/data.ntrans_0.1.nitems_0.1.1'
    p, p_, count = read_file(filename)
    #p, p_, count = read_outfile(filename2)
    #p, p_, count = read_outfile_bi(filename2)
    a, x = create_matrix(p, p_, count)
    tStart = time.time()
    r, ite = pagerank(a, x)
    tEnd = time.time()
    output(r, count, tStart, tEnd, ite)
