# CUDA_VISIBLE_DEVICES='0' python gan.py
import argparse
import struct
import time
import random
import numpy as np ; print('numpy ' + np.__version__)
import tensorflow as tf ; print('tensorflow ' + tf.__version__)
from multiprocessing import Process, Queue
import lifelib ; print('lifelib',lifelib.__version__)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--net', help='network arch', default='dense')
parser.add_argument('--layers', help='number of layers', default=5, type=int)
parser.add_argument('--units', help='number of units/layer', default=1000, type=int)
parser.add_argument('--filters', help='number of filters/layer', default=256, type=int)
parser.add_argument('--lr', help='learning rate', default=0.00001, type=float)
parser.add_argument('--threads', help='number of threads', default=40, type=int)
parser.add_argument('--batch', help='batch size', default=1000, type=int)
parser.add_argument('--bpe', help='batches per epoch', default=1000, type=int)
parser.add_argument('--epochs', help='training epochs', default=10000000, type=int)
parser.add_argument('--model', default='default.proto')
parser.add_argument('--debug', default=False, action='store_true')
args = parser.parse_args()
print(args)

np.set_printoptions(linewidth=250)
np.set_printoptions(formatter={'float': '{:12.8f}'.format, 'int': '{:4d}'.format})

def dnet_dense(args,x,reuse=None):
    print('discriminator network, reuse',reuse)
    with tf.variable_scope('d',reuse=reuse):
        d = tf.identity(x) ; print(d)
        d = tf.layers.flatten(inputs=d) ; print(d)
        for i in range(args.layers):
            d = tf.layers.dense(inputs=d, units=args.units, activation=tf.nn.selu) ; print(d)
        d = tf.layers.dense(inputs=d, units=500, activation=None) ; print(d)
    return d

def dnet_conv(args,x,reuse=None):
    print('discriminator network, reuse',reuse)
    with tf.variable_scope('d',reuse=reuse):
        d = tf.identity(x) ; print(d)
        for i in range(args.layers):
            d = tf.layers.conv2d(inputs=d, filters=args.filters, kernel_size=3, strides=1,activation=tf.nn.selu, padding='valid') ; print(d)
        #d = tf.layers.conv2d(inputs=d, filters=1, kernel_size=3, strides=1,activation=tf.nn.selu, padding='valid') ; print(d)
        d = tf.layers.flatten(inputs=d) ; print(d)
        d = tf.layers.dense(inputs=d, units=args.units, activation=tf.nn.selu) ; print(d)
        d = tf.layers.dense(inputs=d, units=args.units, activation=tf.nn.selu) ; print(d)
        d = tf.layers.dense(inputs=d, units=500, activation=None) ; print(d)
    return d

x = tf.placeholder(tf.float32, [None,16,16,1],name='x') ; print(x)
y = tf.placeholder(tf.int32,(None),name='y') ; print(y) # [batch] label

logits = locals()['dnet_'+args.net](args,x)
loss = tf.losses.sparse_softmax_cross_entropy(labels=y,logits=logits) ; print(loss)
pred = tf.nn.softmax(logits,name='pred') ; print(pred)

#opt = tf.train.GradientDescentOptimizer(learning_rate=args.lr)
opt = tf.train.AdamOptimizer(learning_rate=args.lr)
grads = opt.compute_gradients(loss)
train = opt.apply_gradients(grads)
norm = tf.global_norm([i[0] for i in grads])
init = tf.variables_initializer(tf.global_variables())

# random symmetry = [p,np.flip(p,axis=1),np.flip(p,axis=2),np.rot90(p,k=1,axes=(1,2)),np.rot90(p,k=2,axes=(1,2)),np.rot90(p,k=3,axes=(1,2))]
def permute(d):
    r = np.random.randint(6)
    if r==0:
        return(d)
    if r==1:
        return(np.flip(d,axis=1))
    if r==2:
        return(np.flip(d,axis=2))
    if r==3:
        return(np.rot90(d,k=1,axes=(1,2)))
    if r==4:
        return(np.rot90(d,k=2,axes=(1,2)))
    if r==5:
        return(np.rot90(d,k=3,axes=(1,2)))
 
def search(args,q,k): # generate args.batch size batches of data and push to queue
    sess = lifelib.load_rules("b3s23")
    d = np.zeros([args.batch,16,16,1],dtype=np.uint8) # pattern
    l = np.zeros([args.batch],dtype=np.uint32) # lifespan
    while True:
        t=0 # total soups in batch
        lt = sess.lifetree(memory=100000)
        while t < args.batch:
            # generate random 16x16 soup
            rle=''
            for i in range(16):
                for j in range(16):
                    b = random.choice([0,1])
                    d[t,i,j,0] = b
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
                    break
                last = h.population
        #print('k',k,l[0:10])
        q.put((d.copy(),l.copy()))

q = Queue(maxsize=1000)
for k in range(args.threads):
    p = Process(target=search,args=(args,q,k,))
    p.daemon = True
    p.start()

with tf.Session() as sess:
    sess.run(init)
    for i in range(args.epochs):
        t0 = time.time()

        # TRAIN
        la=[]
        ga=[]
        for j in range(args.bpe):
            (d,l) = q.get(block=True,timeout=None)
            # TODO add symmetries
            d = np.concatenate([d,np.flip(d,axis=1),np.flip(d,axis=2),np.rot90(d,k=1,axes=(1,2)),np.rot90(d,k=2,axes=(1,2)),np.rot90(d,k=3,axes=(1,2))],axis=0)
            l = np.concatenate([l,l,l,l,l,l],axis=0)
            _,loss_,grad_ = sess.run([train,loss,norm],feed_dict={x:d,y:l})
            la.append(loss_)
            ga.append(grad_)

        # TEST
        aa=[]
        for j in range(100):
            (d,l) = q.get(block=True,timeout=None)
            p = sess.run(pred, feed_dict={x:d})
            aa.append(np.mean(np.argmax(p, axis=1) == l))
        
        tf.train.write_graph(tf.graph_util.convert_variables_to_constants(sess, tf.get_default_graph().as_graph_def(), ['pred']), '.', args.model, as_text=False)
        t1 = time.time()
        print('trainops {:9d} t1 {:6.2f} loss {:12.8f} grad {:12.8f} accuracy {:12.8f}'.format(i*args.bpe,t1-t0,np.mean(la),np.mean(ga),np.mean(aa)))