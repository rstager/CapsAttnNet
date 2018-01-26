"""
Evaluate a previously trained CAN model

Usage:
       python eval.py
       ... ...

Result:

"""
import numpy as np


from keras.models import load_model
import os

from keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
from canlayer import CAN,PrimaryCap,dim_geom
from itertools import product

from train import cached_onehot_generators, create_model, margin_loss, pose_loss
from gen_images import default_objects as objs

def print_W1(w1):

    shape=w1.shape
    for i0,i1,i2 in product(range(shape[0]), range(shape[1]), range(shape[2])):
        print()
        print("icap {}, ncap {}, npart {}".format(i0,i1,i2))
        for i3 in range(shape[3]):
            line=""
            for i4 in range(shape[4]):
                line+="{: 0.3f} ".format(w1[i0,i1,i2,i3,i4])
            print(line)

def print_contributions(w1,x,label):
    # W1.shape =[input_num_capsule, num_capsule, num_part, dim_geom + 1, dim_geom]
    shape=w1.shape
    for i0,i1,i2 in product(range(shape[0]), range(shape[1]), range(shape[2])):
        print(" {: 1.3f} {: 1.3f} {: 1.3f} {: 1.3f} {: 1.3f} {: 1.3f}".format(
            w1[i0, i1, i2, 6, 0], w1[i0, i1, i2, 6, 1],w1[i0,i1,i2,0,0],
            w1[i0,i1,i2,0,1],w1[i0,i1,i2,1,0],w1[i0,i1,i2,1,1]))

def eval(model,test_gen,args):
    w1=model.get_layer('can_1').get_weights()[0]
    print_W1(w1)

    ncol=4
    nrow=int((args.num-1)/ncol)+1
    idx=0
    f, axarr = plt.subplots(nrow,ncol)
    plt.tight_layout()
    while True:
        X,Y_true=next(test_gen)
        Y_pred=model.predict(X)
        for x,y_true,y_pred,pose_true,pose_pred in zip(X,Y_true[0],Y_pred[0],Y_true[1],Y_pred[1]):
            ax=axarr[int(idx/ncol),idx%ncol]
            ax.set_axis_off()

            ax.imshow(np.squeeze(x,2),extent=[-.5,.5,-.5,.5])

            label_true=np.argmax(y_true[0])
            label_pred=np.argmax(y_pred[0])
            probability=y_pred[0,label_pred]

            ax.scatter(pose_true[label_pred,0,0:1],pose_true[label_pred,0,1:2])
            ax.scatter(pose_pred[label_pred,0,0:1],pose_pred[label_pred,0,1:2])

            ax.set_title("{} {:.2f}".format(objs[label_pred][0],probability))

            if True:
                line1, line2 = "", ""
                for p,t in zip( pose_true[label_true, 0],pose_pred[label_pred, 0]):
                    line1 += "{: 0.3f} ".format(p)
                    line2 += "{: 0.3f} ".format(t)
                print()
                print("true {:10} {}".format(objs[label_true][0],line1) )
                print("pred {:10} {}".format(objs[label_pred][0],line2))
                #print_contributions(w1,pose_pred[label_pred,0],label_pred)

            idx +=1
            if (idx == args.num):
                plt.pause(60)
                idx=0
                return

if __name__ == "__main__":
    import os
    import argparse
    from keras.preprocessing.image import ImageDataGenerator
    from keras import callbacks

    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="Capsule Network on MNIST.")
    parser.add_argument('--num', default=16, type=int)
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('--file', default='images',
                        help="filename for cached images.. see gen_images")
    parser.add_argument('--data', default="./data/",
                        help="data directory for cached images")
    parser.add_argument('--model', default="trained_model.h5",
                        help="model filename")
    args = parser.parse_args()

    # it might be nice to support non-file generators, but this seems to run faster
    train_gen, test_gen = cached_onehot_generators(args.data,args.file)
    print(dim_geom)
    model = load_model(os.path.join(args.save_dir,args.model),
                       custom_objects={
                           'CAN':CAN,
                           'PrimaryCap':PrimaryCap,
                           'margin_loss':margin_loss,
                           'pose_loss':pose_loss,

                       })

    model.summary()


    eval(model, test_gen, args=args)

