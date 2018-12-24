import numpy as np
import time

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
    w = [np.zeros(count) for i in range(count)]
    i_ = np.identity(count)
    w = np.array(w)
    for i in range(len(p)):
        w[p[i]-1][p_[i]-1] = 1
    w = w.T
    for i in range(count):
        if (np.sum(w[i]) != 0):
            w[i] = w[i] / np.sum(w[i])
    w = w.T
    s = (1 - 0.8) * i_
    return w, s, i_

def simrank(w, s, i_):
    e = 1
    ite = 0
    while (e > 0.0001):
        past_s = s
        s = 0.8 * np.dot(np.dot(w.T, s),w) + i_ -np.diag(np.diag(0.8 * np.dot(np.dot(w.T, s),w)))
        e = np.linalg.norm(s-past_s)
        ite = ite + 1
    return s, ite

def output(s, count, tStart, tEnd, ite):
    print('Sim rank')
    print(s)
    print('count:',count,'   executing simrank function time:','{:08e}'.format( - tStart + tEnd ),'   simrank function iteration counts:',ite)

    
    
if __name__ == "__main__":
    filename = 'hw3dataset/graph_5.txt'
    filename2 = 'hw3dataset/data.ntrans_0.1.nitems_0.1.1'
    p, p_, count = read_file(filename)
    #p, p_, count = read_outfile(filename2)
    #p, p_, count = read_outfile_bi(filename2)
    w, s, i_ = create_matrix(p, p_, count)
    tStart = time.time()
    s, ite = simrank(w, s, i_)
    tEnd = time.time()
    output(s, count, tStart, tEnd, ite)
