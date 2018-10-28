import argparse
import numpy as np ; print 'numpy ' + np.__version__
import cv2 ; print 'cv2 ' + cv2.__version__
import pickle
import serial ; print 'serial ' + serial.VERSION
import time
import os

# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--serial', help='serial port',default='/dev/ttyUSB0')
parser.add_argument('--dir', help='output directory',default='./save/test')
parser.add_argument('--size', help='size of life world', default=600, type=int)
parser.add_argument('--init', help='size of init patch', default=20, type=int)
parser.add_argument('--rngmode', help='random init distribution 00=0.5, 10=0.25, 01=0.75, 11=cycle',default='11')
parser.add_argument('--num_init', help='num_init limit for fpga', default=1000000, type=int)
#parser.add_argument('--act_thresh', help='minimum activity threshold for boundary check', default=1000, type=int)
parser.add_argument('--verbose', default=False, action='store_true')
parser.add_argument('--nocheck', default=False, action='store_true')
parser.add_argument('--debug', default=False, action='store_true')
args = parser.parse_args()
print(args)

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

SERIALPORT = args.serial
BAUDRATE = 115200
ser = serial.Serial(SERIALPORT,BAUDRATE,exclusive=True)
ser.cancel_read()
#ser.cancel_write()
if ser.is_open:
    try:
        while True:
            ser.send_break();
            time.sleep(0.5)
            ser.reset_input_buffer() #flush input buffer, discarding all its contents
            ser.reset_output_buffer()#flush output buffer, aborting current output
            if args.verbose:
                print 'time {}'.format(time.strftime("%Y%m%d-%H%M%S"))
            #cmd = 'R' + format(args.act_thresh, '032b')[::-1] + format(args.num_init, '032b')[::-1] + args.rngmode
            cmd = 'R' + format(args.num_init, '032b')[::-1] + args.rngmode
            if args.verbose:
                print 'cmd',cmd
            for c in cmd:
                ser.write(c)
                time.sleep(0.1)
            response = ser.read()
            if response=='L':
                init=''
                for i in range(args.init*args.init):
                    init += ser.read()
                step_count = int(ser.read(32)[::-1],2)
                #boundact = int(ser.read(32)[::-1],2)
                if args.verbose:
                    print 'step_count',step_count
                w = np.zeros(args.size*args.size,dtype=np.int)
                winit = np.zeros(args.init*args.init,dtype=np.int)
                for j in range(args.init*args.init):
                    if init[j]=='0':
                        w[(((args.size/2)-(args.init/2)+(j/args.init))*args.size)+(args.size/2)-(args.init/2)+(j%args.init)] = 0; # center
                        winit[j]=0
                    else:
                        w[(((args.size/2)-(args.init/2)+(j/args.init))*args.size)+(args.size/2)-(args.init/2)+(j%args.init)] = 1; # center
                        winit[j]=1
                w = np.reshape(w,(args.size,args.size),order='C')
                winit = np.reshape(winit,(args.init,args.init),order='C')
                if not args.nocheck:
                    if args.verbose:
                        print 'start run',time.strftime("%Y%m%d%H%M%S")
                    pcmd = './fastrun -c %s'%(init)
                    if args.verbose:
                        print 'popen',pcmd
                    final = os.popen(pcmd).read()
                    final = int(final)
                    if args.verbose:
                        print 'end run',time.strftime("%Y%m%d%H%M%S")
                else:   
                    final=step_count

                #fn = args.dir+'/F'+'%05d'%final+'-L'+'%05d'%step_count+'-A'+'%09d'%boundact+'-R'+args.rngmode+'-T'+time.strftime("%Y_%m%d_%H%M_%S")
                fn = args.dir+'L'+'%05d'%step_count+'-S'+'%09d'%args.num_init+'-R'+args.rngmode+'-T'+time.strftime("%Y_%m%d_%H%M_%S")
                write_lif(winit,fn+'.lif',args)
                print 'step_count {:9,d} time {} rng {} init0 {:4d} init1 {:4d} fn {}'.format(step_count,time.strftime("%Y_%m%d_%H%M_%S"),args.rngmode,init.count('0'),init.count('1'),fn+'.lif')
            else:
                print 'error: response not L'
    except Exception, e:
        print "error communicating...: " + str(e)
else:
    print "cannot open serial port "
