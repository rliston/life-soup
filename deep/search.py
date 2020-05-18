import lifelib ; print('lifelib',lifelib.__version__)
import random
import numpy as np ; print('numpy ' + np.__version__)
import argparse
import time
import threading

np.set_printoptions(linewidth=250)
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--threads', help='number of threads', default=40, type=int)
parser.add_argument('--n', help='number of soups per thread', default=10000000, type=int)
parser.add_argument('--npz', help='output npz data file name base', default='soup_')
parser.add_argument('--interval', help='save data file every interval soups', default=1000000, type=int)
parser.add_argument('--verbose', default=False, action='store_true')
parser.add_argument('--debug', default=False, action='store_true')
args = parser.parse_args()
print(args)

def search(args,k): # generate args.npz+str(k)+'.npz' with args.n soups
    sess = lifelib.load_rules("b3s23")
    t0 = time.time()
    t1 = time.time()
    d = np.zeros([args.n,8],dtype=np.uint32) # pattern, packed bits
    l = np.zeros([args.n],dtype=np.uint32) # lifespan

    t=0 # total soups
    while t < args.n:
        lt = sess.lifetree(memory=100000)

        # generate random 16x16 soup
        rle=''
        for i in range(16):
            for j in range(16):
                b = random.choice([0,1])
                d[t,i>>1] |= (b<<(j+(i&1)))
                rle += ['b','o'][b]
            rle += '$'
        rle = rle[:-1] + '!'
        h = lt.pattern(rle)

        # run soup until population is stable
        last=None
        for i in range(1000):
            h = h.advance(100)
            if h.population == last:
                l[t] = i
                t += 1
                if t%args.interval==0:
                    np.savez('{}{%3d}.npz'.format(args.npz,k),pattern=d[0:t],lifespan=l[0:t])
                    b = np.arange(0,120,10) # 0 to 10K+ in 1K steps
                    hist = np.histogram(np.clip(l[0:t],b[0],b[-1]),bins=b)[0] # put >10K in last bin
                    print('k {:3d} t {:9d} progress {:6.4f} t0 {:8.2f} t1 {:8.2f} histogram {}'.format(k,t,t/args.n,time.time()-t0,time.time()-t1,hist))
                    t1 = time.time()
                break
            last = h.population

for k in range(args.threads):
    s = threading.Thread(target=search, args=(args,k)
    s.daemon=True
    s.start()

