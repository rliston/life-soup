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
parser.add_argument('--size',help='input size (16 for dense, 32 for conv)',default=16,type=int)
parser.add_argument('--depth',help='input depth (1 for dense, 9 for conv)',default=1,type=int)
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

def gensoup(lt,size=16,depth=1):
    mat2pat = lambda M : lt.pattern('$'.join([''.join(['bo'[y] for y in x]) for x in M.tolist()]) + '!')
    coords = np.array([[x,y] for x in range(size) for y in range(size)],dtype=np.int64)

    d = np.zeros([size,size,depth],dtype=np.uint8) # pattern
    o = depth-1
    s = np.random.randint(2,size=(16,16)) # layer 0 = centered random 16x16 soup

    rle=''
    for i in range(16):
        for j in range(16):
            rle += ['b','o'][s[i,j]]
        rle += '$'
    rle = rle[:-1] + '!'

    d[o:o+16,o:o+16,0] = s # layer 0 = centered random 16x16 soup
    p = mat2pat(d[:,:,0])
    #rle = p.rle_string().replace('\n', ' ') # save initial soup
    for k in range(1,depth):
        p = p.advance(1)
        v = p[coords].reshape((size,size),order='F')
        d[:,:,k] = v
    return p,d,rle

def stabilize(pat):
    for i in range(1000):
        pop = pat.population
        pat = pat.advance(100)
        if pat.population == pop:
            pat = pat.advance(2)
            if pat.population == pop:
                pat = pat.advance(2)
                if pat.population == pop:
                    return i*100
    return -1


# load model
sess = tf.Session()
with open(args.model, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='foo') # name the model in case we ever need to load multiple models

r = []
lsess = lifelib.load_rules("b3s23")
for k in range(args.n):
    # generate random soup
    lt = lsess.lifetree(memory=100000)
    p,d,rle = gensoup(lt,args.size,args.depth)
    life = stabilize(p)

    # run deep model
    pdf = sess.run('foo/pred:0', feed_dict={'foo/x:0':[d]})[0]
    pred = np.argmax(pdf)
    prob = pdf[pred]

    # (rle,pred,prob,life)
    print('life {:6d} pred {:6d} prob {:12.8f} rle {}'.format(life,pred**2,prob,rle))
    r.append((rle,life,pred,prob))
