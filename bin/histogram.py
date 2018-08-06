import argparse
import numpy as np ; print 'numpy ' + np.__version__
import cv2 ; print 'cv2 ' + cv2.__version__
import pickle
import serial ; print 'serial ' + serial.VERSION
import time
import os
import matplotlib.pyplot as plt

# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--samples', help='number of random inits to generate for histogram', default=1000000, type=int)
parser.add_argument('--size', help='size of life world', default=600, type=int)
parser.add_argument('--init', help='size of init patch', default=20, type=int)
parser.add_argument('--pickle', help='histogram pickle array',default='histogram.pickle')
parser.add_argument('--vis', default=False, action='store_true')
#parser.add_argument('--lif', help='initial seed lif file',default='')
#parser.add_argument('--scale', help='visualization scale factor', default=3, type=int)
#parser.add_argument('--run', default=False, action='store_true')
#parser.add_argument('--act', default=False, action='store_true')
#parser.add_argument('--max', default=False, action='store_true')
#parser.add_argument('--init', help='size of init patch', default=50, type=int)
#parser.add_argument('--maxsteps', help='max_steps limit for fpga', default=5000, type=int)
#parser.add_argument('--boundact', help='minimum activity threshold for boundary check', default=10, type=int)
#parser.add_argument('--maxact', help='max_act limit for fpga', default=10000000, type=int)
#parser.add_argument('--maxinit', help='max_init limit for fpga', default=100000000, type=int)
#parser.add_argument('--rngmode', help='random init distribution 00=0.5, 10=0.25, 01=0.75, 11=cycle',default='00')
#parser.add_argument('--speriod0', help='sample period 0 for repeat check', default=13, type=int)
#parser.add_argument('--speriod1', help='sample period 1 for repeat check', default=199, type=int)
#parser.add_argument('--speriod2', help='sample period 2 for repeat check', default=997, type=int)
#parser.add_argument('--dir', help='output directory',default='./')
parser.add_argument('--verbose', default=False, action='store_true')
parser.add_argument('--debug', default=False, action='store_true')
args = parser.parse_args()
print(args)

def visualize(w,scale,r,g,b):
    img=np.zeros([args.size,args.size,3],dtype=np.uint8)
    img[:,:,0] = w*b
    img[:,:,1] = w*g
    img[:,:,2] = w*r
    #img = cv2.resize(img,None,fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)
    img = cv2.resize(img,None,fx=scale, fy=scale, interpolation = cv2.INTER_NEAREST)
    #img = cv2.resize(img,None,fx=scale, fy=scale, interpolation = cv2.INTER_LANCZOS4)
    cv2.imshow('img',img)

def init(args):
    u = np.random.binomial(1,0.5,size=[args.init,args.init])
    v = np.zeros([args.size,args.size])
    x = int(args.size*0.5-args.init*0.5)
    v[x:x+args.init,x:x+args.init] = u
    return u,v

def step(w):
    w0 = np.roll(w,1,axis=0) ; w0[0,:]=0 # down y
    w1 = np.roll(w,-1,axis=0) ; w1[-1,:]=0 # up y
    w2 = np.roll(w,1,axis=1) ; w2[:,-1]=0 # right x
    w3 = np.roll(w,-1,axis=1) ; w3[:,0]=0 # left x
    w4 = np.roll(w,1,axis=0) ; w4[0,:]=0 ; w4 = np.roll(w4,1,axis=1) ; w4[:,-1]=0
    w5 = np.roll(w,-1,axis=0) ; w5[-1,:]=0 ; w5 = np.roll(w5,1,axis=1) ; w5[:,-1]=0
    w6 = np.roll(w,-1,axis=0) ; w6[-1,:]=0 ; w6 = np.roll(w6,-1,axis=1) ; w6[:,0]=0
    w7 = np.roll(w,1,axis=0) ; w7[0,:]=0 ; w7 = np.roll(w7,-1,axis=1) ; w7[:,0]=0

    s = w0+w1+w2+w3+w4+w5+w6+w7
    w = np.logical_or(np.logical_and(w, s==2), s==3)
    w = w.astype(np.int)
    return w

def run(w):
    h={}
    for i in range(100000): # max 100000 steps
        k=hash(w.tostring())
        if k in h:
            return i
        h[k]=1
        w = step(w)
    return i

def write_lif(w,fn,args):
    f = open(fn, 'w')
    print >>f, '#Life 1.05'
    print >>f, '#D ' + fn
    print >>f, '#P ' + str(-int(args.init)*0.5) + ' ' + str(-int(args.init)*0.5)
    for i in range(w.shape[0]):
        for j in range(w.shape[1]):
            if w[i,j]==0:
                f.write('.')
            else:
                f.write('*')
        f.write('\n')
    f.close()

if args.vis:
    hdat = pickle.load(open(args.pickle, 'rb'))
    print 'len(hdat)',len(hdat)
    h=[]
    for t in hdat:
        if t[0] < 1100:
            h.append(t[0])
        else:
            h.append(t[0]-1050) # roughly compensate for glider escapes
    h = np.array(h)
    plt.hist(h,bins=200, density=False, facecolor='green')
    plt.xlim(0, 40000)
    plt.minorticks_on()
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.show()
    exit()

### generate args.samples random inits, generate histogram of num_steps
if os.path.isfile(args.pickle):
    hdat = pickle.load(open(args.pickle, 'rb'))
else:
    hdat=[]
for i in range(args.samples):
    u,w = init(args)
    s = run(w)
    hdat.append([s,u])
    if args.verbose:
        print 'i',i,'s',s,'len(hdat)',len(hdat)
    fn = '.'+'/L'+'%05d'%s+'.lif'
    write_lif(u,fn,args)
    pickle.dump(hdat, open(args.pickle,'wb'), protocol=pickle.HIGHEST_PROTOCOL)
