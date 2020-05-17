import argparse
import time
import numpy as np ; print('numpy ' + np.__version__)
import tensorflow as tf ; print('tensorflow ' + tf.__version__)
import matplotlib.pyplot as plt
import random
import lifelib ; print('lifelib',lifelib.__version__)

# parse command line
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#parser.add_argument('--fft', default=False, action='store_true') # output logging messages to stderr
parser.add_argument('--model', default='default.proto')
#parser.add_argument('--npy', help='npy data file from merge.py', default='default.npy')
parser.add_argument('--n', help='number of soups', default=10, type=int)
#parser.add_argument('--shuffle', default=False, action='store_true') # output logging messages to stderr
#parser.add_argument('--freq',help='uhd center frequency',default=5180000000,type=float)
#parser.add_argument('--rx_gain',help='normalized uhd rx gain',default=1.0,type=float)
#parser.add_argument('--snr',help='automatically estimate noise floor and set SNR threshold for frame capture',default=15.0,type=float)
#parser.add_argument('--serial', help='usrp b200 serial number', default=None)
#parser.add_argument('--antenna', help='usrp antenna (RX2, TX/RX)', default='TX/RX')
#parser.add_argument('--cpu_format', help='usrp cpu_format {fc32,sc16}', default='fc32')
#parser.add_argument('--nchannels',help='number of uhd channels',default=1,type=int)
#parser.add_argument('--samp_rate',help='uhd sample rate',default=20000000,type=float)
#parser.add_argument('--window_size',help='number of samples used to estimate signal power eg. 4us',default=80,type=int)
#parser.add_argument('--max_windows',help='maximum number of sample windows to capture per packet',default=10000,type=int) # this affect the noise floor estimate
#parser.add_argument('--ipg',help='number of sample windows to capture before and after packet',default=1,type=int)
#parser.add_argument('--npkts',help='number of packets to capture',default=1000000000,type=int)
#parser.add_argument('--nsecs',help='number of seconds to capture',default=10,type=int)
#parser.add_argument('--verbose', default=False, action='store_true') # output logging messages to stderr
#parser.add_argument('--debug', default=False, action='store_true') # output logging messages to stderr
#parser.add_argument('--agc', default=False, action='store_true') # reduce rx_gain if clipping detected, experimental
args = parser.parse_args()
print(args)

# load model
sess = tf.Session()
with open(args.model, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='foo') # name the model in case we ever need to load multiple models

r = []
lsess = lifelib.load_rules("b3s23")
for k in range(args.n):
    # generate random 16x16 soup
    p = np.zeros([16,16],dtype=np.uint8) # pattern
    rle=''
    for i in range(16):
        for j in range(16):
            p[i,j] = random.choice([0,1])
            rle += ['b','o'][p[i,j]]
        rle += '$'
    rle = rle[:-1] + '!'

    # run deep model
    pdf = sess.run('foo/pred:0', feed_dict={'foo/x:0':p.reshape([1,16,16,1])})[0]
    pred = np.argmax(pdf)
    prob = pdf[pred]

    # run soup until population is stable
    lt = lsess.lifetree(memory=100000)
    h = lt.pattern(rle)
    last=None
    life=None
    for i in range(1000):
        h = h.advance(100)
        if h.population == last:
            life=i
            break
        last = h.population
    
    # (rle,pred,prob,life)
    print(rle,'life',life,'pred',pred,'prob',prob)
    r.append((rle,life,pred,prob))

#            if i > args.m:
#                if args.verbose:
#                    print('k',k,'l',i,'pop',h.population,'rle',rle)
#                l = i
#                k += 1
#                if k%1000==0:
#                    np.savez(args.npz,pattern=p[0:k],lifespan=l[0:k])
#                    b = np.arange(0,120,10) # 0 to 10K+ in 1K steps
#                    h = np.histogram(np.clip(l[0:k],b[0],b[-1]),bins=b)[0]
#                    print('k {:9d} tot {:9d} ({:6.4f}) progress {:6.4f} t0 {:8.2f} t1 {:8.2f} histogram {}'.format(k,t,k/t,k/args.n,time.time()-t0,time.time()-t1,h))
#                    t1 = time.time()
#            break
#v=[]
#for k in range(args.n):
#    #print('k',k,'pred',np.argmax(pred[k]),pred[k,np.argmax(pred[k])],'rle',r[k])
#    v.append([np.argmax(pred[k]),pred[k,np.argmax(pred[k])],r[k]])
#
#v = sorted(v, key=lambda x:x[1])
#for k in range(args.n):
#    print(v[k])
