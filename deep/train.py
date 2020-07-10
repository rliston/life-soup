# CUDA_VISIBLE_DEVICES='0' unbuffer python train.py |& tee default.log
# TODO
# 1. use tf.Model class, tf.saved_model, output logits in order to fine tune pretrained model
# 2. in training loop, check if queue.empty() or np.mean(queue.qsize()) ... for performance monitoring

import argparse
import struct
import time
import random
import numpy as np ; print('numpy ' + np.__version__)
import tensorflow as tf ; print('tensorflow ' + tf.__version__)
from multiprocessing import Process, Queue
import lifelib ; print('lifelib',lifelib.__version__)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--stab', help='stabilization step', default=6, type=int)
parser.add_argument('--net', help='network arch', default='conv')
parser.add_argument('--clayers', help='number of convolutional layers', default=16, type=int)
parser.add_argument('--layers', help='number of layers', default=18, type=int)
parser.add_argument('--units', help='number of units/layer (dense)', default=3000, type=int)
parser.add_argument('--filters', help='number of filters/layer (conv)', default=32, type=int)
parser.add_argument('--lr', help='learning rate', default=0.000001, type=float)
parser.add_argument('--pow', help='label weight power exponent', default=0., type=float)
parser.add_argument('--threads', help='number of threads for generating soups', default=20, type=int)
parser.add_argument('--batch', help='batch size', default=30, type=int)
parser.add_argument('--bpe', help='batches per epoch', default=10000, type=int)
parser.add_argument('--epochs', help='training epochs', default=1000000000, type=int)
parser.add_argument('--model', default='default.proto')
parser.add_argument('--debug', default=False, action='store_true')
args = parser.parse_args()
print(args)

np.set_printoptions(linewidth=250)
np.set_printoptions(formatter={'float': '{:12.8f}'.format, 'int': '{:4d}'.format})

if args.net=='dense':
    args.size=16
    args.depth=1
    x = tf.placeholder(tf.float32, [None,args.size,args.size,args.dense],name='x') ; print(x)
    y = tf.placeholder(tf.int32,(None),name='y') ; print(y) # [batch] label
    with tf.variable_scope('d',reuse=None):
        d = tf.identity(x) ; print(d)
        d = tf.layers.flatten(inputs=d) ; print(d)
        for i in range(args.layers):
            d = tf.layers.dense(inputs=d, units=args.units, activation=tf.nn.selu) ; print(d)
        d = tf.layers.dense(inputs=d, units=200, activation=None) ; print(d)
    logits = tf.identity(d) ; print(logits)
 
if args.net=='conv':
    args.size=32 # start with centered 16x16, run for 8 steps and input as convolutional layers
    args.depth=9
    x = tf.placeholder(tf.float32, [None,args.size,args.size,args.depth],name='x') ; print(x)
    y = tf.placeholder(tf.int32,(None),name='y') ; print(y) # [batch] label
    with tf.variable_scope('d',reuse=None):
        d = tf.identity(x) ; print(d)
        for i in range(1,args.clayers):
            d = tf.layers.conv2d(inputs=d, filters=args.filters*i, kernel_size=3, strides=1,activation=tf.nn.selu, padding='valid') ; print(d)
        #d = tf.layers.conv2d(inputs=d, filters=4, kernel_size=3, strides=1,activation=tf.nn.selu, padding='valid') ; print(d)
        d = tf.layers.flatten(inputs=d) ; print(d)
        for i in range(args.layers):
            d = tf.layers.dense(inputs=d, units=args.units, activation=tf.nn.selu) ; print(d)
        d = tf.layers.dense(inputs=d, units=200, activation=None) ; print(d)
    logits = tf.identity(d) ; print(logits)
 
if args.net=='conv2':
    args.size=32 # start with centered 16x16, run for 8 steps and input as convolutional layers
    args.depth=9
    x = tf.placeholder(tf.float32, [None,args.size,args.size,args.depth],name='x') ; print(x)
    y = tf.placeholder(tf.int32,(None),name='y') ; print(y) # [batch] label
    with tf.variable_scope('d',reuse=None):
        d = tf.identity(x) ; print(d)
        for i in range(1,16):
            d = tf.layers.conv2d(inputs=d, filters=args.filters*i, kernel_size=3, strides=1,activation=tf.nn.relu, padding='valid') ; print(d)
        d = tf.layers.conv2d(inputs=d, filters=2, kernel_size=2, strides=1,activation=tf.nn.relu, padding='valid') ; print(d)
        d = tf.layers.flatten(inputs=d) ; print(d)
        d = tf.layers.dense(inputs=d, units=200, activation=None) ; print(d)
    logits = tf.identity(d) ; print(logits)
 
if args.net=='sig':
    args.size=64 # start with centered 16x16, run for 24 steps and input as convolutional layers
    args.depth=25
    x = tf.placeholder(tf.float32, [None,args.size,args.size,args.depth],name='x') ; print(x)
    y = tf.placeholder(tf.int32,(None),name='y') ; print(y) # [batch] label
    with tf.variable_scope('d',reuse=None):
        d = tf.identity(x) ; print(d)
        for i in range(1,32):
            d = tf.layers.conv2d(inputs=d, filters=args.filters*i, kernel_size=3, strides=1,activation=tf.sigmoid, padding='valid') ; print(d)
        d = tf.layers.flatten(inputs=d) ; print(d)
        for i in range(args.layers):
            d = tf.layers.dense(inputs=d, units=args.units, activation=tf.sigmoid) ; print(d)
        d = tf.layers.dense(inputs=d, units=200, activation=None) ; print(d)
    logits = tf.identity(d) ; print(logits)
 
if args.net=='big':
    args.size=64 # start with centered 16x16, run for 24 steps and input as convolutional layers
    args.depth=25
    x = tf.placeholder(tf.float32, [None,args.size,args.size,args.depth],name='x') ; print(x)
    y = tf.placeholder(tf.int32,(None),name='y') ; print(y) # [batch] label
    with tf.variable_scope('d',reuse=None):
        d = tf.identity(x) ; print(d)
        for i in range(1,32):
            d = tf.layers.conv2d(inputs=d, filters=args.filters*i, kernel_size=3, strides=1,activation=tf.nn.selu, padding='valid') ; print(d)
        #d = tf.layers.conv2d(inputs=d, filters=10, kernel_size=2, strides=1,activation=tf.nn.selu, padding='valid') ; print(d)
        d = tf.layers.flatten(inputs=d) ; print(d)
        for i in range(args.layers):
            d = tf.layers.dense(inputs=d, units=args.units, activation=tf.nn.selu) ; print(d)
        d = tf.layers.dense(inputs=d, units=200, activation=None) ; print(d)
    logits = tf.identity(d) ; print(logits)
 
loss = tf.losses.sparse_softmax_cross_entropy(labels=y,logits=logits,weights=tf.pow(tf.cast(y,tf.float32),tf.constant(args.pow))) ; print(loss)
pred = tf.nn.softmax(logits,name='pred') ; print(pred)

#opt = tf.train.GradientDescentOptimizer(learning_rate=args.lr)
opt = tf.train.AdamOptimizer(learning_rate=args.lr)
grads = opt.compute_gradients(loss)
train = opt.apply_gradients(grads)
norm = tf.global_norm([i[0] for i in grads])
init = tf.variables_initializer(tf.global_variables())

def stabilize(pat):
    life=0
    for i in range(int(100000/args.stab)):
        pop = pat.population
        pat = pat.advance(args.stab)
        life += args.stab
        if pat.population == pop:
            pat = pat.advance(6)
            life += 6
            if pat.population == pop:
                pat = pat.advance(6)
                life += 6
                if pat.population == pop:
                    pat = pat.advance(6)
                    life += 6
                    if pat.population == pop:
                        return life
    return -1

# TODO add         transforms = ["flip", "rot180", "identity", "transpose", "flip_x", "flip_y", "rot90", "rot270", "swap_xy", "swap_xy_flip", "rcw", "rccw"]
def gensoup(lt,size=16,depth=1):
    mat2pat = lambda M : lt.pattern('$'.join([''.join(['bo'[y] for y in x]) for x in M.tolist()]) + '!')
    coords = np.array([[x,y] for x in range(size) for y in range(size)],dtype=np.int64)

    d = np.zeros([size,size,depth],dtype=np.uint8) # pattern
    o = depth-1
    d[o:o+16,o:o+16,0] = np.random.randint(2,size=(16,16)) # layer 0 = centered random 16x16 soup
    p = mat2pat(d[:,:,0])
    for k in range(depth):
        v = p[coords].reshape((size,size),order='F')
        d[:,:,k] = v
        p = p.advance(1)
    return p,d

def genbatch(args,q,k): # generate args.batch size batches of data and push to queue
    sess = lifelib.load_rules("b3s23")
    d = np.zeros([args.batch,args.size,args.size,args.depth],dtype=np.uint8) # pattern
    l = np.zeros([args.batch],dtype=np.uint32) # lifespan
    while True:
        t=0 # total soups in batch
        lt = sess.lifetree(memory=100000)
        while t < args.batch:
            # generate random 16x16 soup
            p,x = gensoup(lt,args.size,args.depth)
            # run pattern until population is stable
            life = stabilize(p)
            if life<0: # didn't stabilize after 100K steps
                continue
            d[t] = x
            l[t] = min(199,int(np.sqrt(life+args.depth-1))) # label = sqrt(lifespan), cap at 199^2=39601
            t+=1
        # generated a batch of (pattern,lifespan)
        q.put((d.copy(),l.copy()))

# IPC
q = Queue(maxsize=1000)
for k in range(args.threads):
    p = Process(target=genbatch,args=(args,q,k,))
    p.daemon = True
    p.start()

with tf.Session() as sess:
    sess.run(init)
    print(args)
    for i in range(1,args.epochs):
        t0 = time.time()

        # TRAIN
        la=[]
        ga=[]
        for j in range(args.bpe):
            (d,l) = q.get(block=True,timeout=None)
            # TODO add symmetries?
            dsym=[]
            lsym=[]
            for flip in range(4):
                for rot in range(4):
                    if flip==0:
                        d0 = d
                    if flip==1:
                        d0 = np.flip(d,axis=1)
                    if flip==2:
                        d0 = np.flip(d,axis=2)
                    if flip==3:
                        d0 = np.flip(d,axis=(1,2))
                    if rot==0:
                        d1 = d0
                    if rot==1:
                        d1 = np.rot90(d0,k=1,axes=(1,2))
                    if rot==2:
                        d1 = np.rot90(d0,k=2,axes=(1,2))
                    if rot==3:
                        d1 = np.rot90(d0,k=3,axes=(1,2))
                    dsym.extend(d1)
                    lsym.extend(l)
            d = np.array(dsym,dtype=np.float)
            l = np.array(lsym,dtype=np.int)
            #d = np.concatenate([d,np.flip(d,axis=1),np.flip(d,axis=2),np.rot90(d,k=1,axes=(1,2)),np.rot90(d,k=2,axes=(1,2)),np.rot90(d,k=3,axes=(1,2))],axis=0)
            #l = np.concatenate([l,l,l,l,l,l],axis=0)
            _,loss_,grad_ = sess.run([train,loss,norm],feed_dict={x:d,y:l})
            la.append(loss_)
            ga.append(grad_)

        tf.train.write_graph(tf.graph_util.convert_variables_to_constants(sess, tf.get_default_graph().as_graph_def(), ['pred']), '.', args.model, as_text=False)
        t1 = time.time()
        print('trainops {:9d} t1 {:6.2f} loss {:12.8f} grad {:12.8f}'.format(i*args.bpe,t1-t0,np.mean(la),np.mean(ga)))
