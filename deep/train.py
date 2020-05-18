# CUDA_VISIBLE_DEVICES='0' python gan.py
import argparse
import struct
import time
import numpy as np
print('numpy ' + np.__version__)
#np.set_printoptions(threshold='nan')
np.set_printoptions(linewidth=250)
np.set_printoptions(formatter={'float': '{:12.8f}'.format, 'int': '{:4d}'.format})
import tensorflow as tf
print('tensorflow ' + tf.__version__)
import cv2
print('cv2 ' + cv2.__version__)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--npz', nargs = '*', dest = 'npz', help = 'npz data files', default = argparse.SUPPRESS)
parser.add_argument('--split', help='train/test ratio', default=0.8, type=float)
parser.add_argument('--net', help='network arch', default='default')
parser.add_argument('--layers', help='number of layers', default=5, type=int)
parser.add_argument('--units', help='number of units/layer', default=1000, type=int)
parser.add_argument('--filters', help='number of filters/layer', default=256, type=int)
parser.add_argument('--lr', help='learning rate', default=0.00001, type=float)
parser.add_argument('--batch', help='batch size', default=1000, type=int)
parser.add_argument('--epochs', help='training epochs', default=10000000, type=int)
parser.add_argument('--model', default='default.proto')
parser.add_argument('--debug', default=False, action='store_true')
args = parser.parse_args()
print(args)

# load meth.npz files from search.py
d=[]
l=[]
for f in args.npz:
    print('loading',f)
    npz = np.load(f)
    d.extend(npz['pattern'])
    l.extend(npz['lifespan'])
d = np.array(d,dtype=np.float32)
l = np.array(l,dtype=np.int32)
d = np.expand_dims(d,axis=-1)
print('d.shape',d.shape,'l.shape',l.shape,'l.min()',l.min(),'l.max()',l.max())

# add symmetries
#d = np.concatenate([p,np.flip(p,axis=1),np.flip(p,axis=2),np.rot90(p,k=1,axes=(1,2)),np.rot90(p,k=2,axes=(1,2)),np.rot90(p,k=3,axes=(1,2))],axis=0)
#l = np.concatenate([l,l,l,l,l,l],axis=0)
#print('d.shape',d.shape,'l.shape',l.shape)

# split train,test
rng = np.random.get_state()
np.random.shuffle(d)
np.random.set_state(rng)
np.random.shuffle(l)
split = int(len(d)*args.split)
td = d[split:]
tl = l[split:]
d = d[0:split]
l = l[0:split]
print('d.shape',d.shape,'l.shape',l.shape)
print('td.shape',td.shape,'tl.shape',tl.shape)

def dnet_default(args,x,reuse=None):
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
        
print('d.shape',d.shape)
with tf.Session() as sess:
    sess.run(init)
    for i in range(args.epochs):
        t0 = time.time()
        rng_state = np.random.get_state()
        np.random.shuffle(d)
        np.random.set_state(rng_state)
        np.random.shuffle(l)

        # TRAIN
        la=[]
        ga=[]
        for j in range(0,d.shape[0],args.batch):
            _,loss_,grad_ = sess.run([train,loss,norm],feed_dict={x:permute(d[j:j+args.batch]),y:l[j:j+args.batch]})
            la.append(loss_)
            ga.append(grad_)

        # TEST
        aa=[]
        for j in range(0,td.shape[0],args.batch):
            p = sess.run(pred, feed_dict={x:permute(td[j:j+args.batch])})
            aa.append(np.mean(np.argmax(p, axis=1) == tl[j:j+args.batch]))
        
        tf.train.write_graph(tf.graph_util.convert_variables_to_constants(sess, tf.get_default_graph().as_graph_def(), ['pred']), '.', args.model, as_text=False)
        t1 = time.time()
        print('epoch {:9d} t1 {:6.2f} loss {:12.8f} grad {:12.8f} accuracy {:12.8f}'.format(i,t1-t0,np.mean(la),np.mean(ga),np.mean(aa)))
