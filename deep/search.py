import lifelib ; print('lifelib',lifelib.__version__)
import random
import numpy as np ; print('numpy ' + np.__version__)
import argparse
import time

np.set_printoptions(linewidth=250)
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--n', help='number of soups', default=1000000, type=int)
parser.add_argument('--npz', help='output npz data file', default='meth.npz')
parser.add_argument('--verbose', default=False, action='store_true')
parser.add_argument('--debug', default=False, action='store_true')
args = parser.parse_args()
print(args)

sess = lifelib.load_rules("b3s23")

p = np.zeros([args.n,16,16],dtype=np.uint8) # pattern
l = np.zeros([args.n],dtype=np.uint32) # lifespan

t=0 # total soups
t0 = time.time()
t1 = time.time()
while t < args.n:
    lt = sess.lifetree(memory=100000)

    # generate random 16x16 soup
    rle=''
    for i in range(16):
        for j in range(16):
            p[t,i,j] = random.choice([0,1])
            rle += ['b','o'][p[t,i,j]]
        rle += '$'
    rle = rle[:-1] + '!'
    h = lt.pattern(rle)

    # run soup until population is stable
    last=None
    for i in range(1000):
        h = h.advance(100)
        #print('i',i,'pop',h.population,'last',last)
        if h.population == last:
            l[t] = i
            t += 1
            if t%1000==0:
                np.savez(args.npz,pattern=p[0:t],lifespan=l[0:t])
                b = np.arange(0,120,10) # 0 to 10K+ in 1K steps
                h = np.histogram(np.clip(l[0:t],b[0],b[-1]),bins=b)[0] # put >10K in last bin
                print('t {:9d} progress {:6.4f} t0 {:8.2f} t1 {:8.2f} histogram {}'.format(t,t/args.n,time.time()-t0,time.time()-t1,h))
                t1 = time.time()
            break
        last = h.population


#print('l.shape',l.shape,'histogram',np.histogram(l),'sum',np.sum(l))
