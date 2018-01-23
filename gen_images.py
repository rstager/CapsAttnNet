'''
image_generator provides a keras generator for images (X) and class+pose(y).
The images contain a single object of a given class.
default_objects defines how to compose the objects from
the graphics primatives (boxes,triangles and circles)

'''

import random
from gym.envs.classic_control.rendering import Transform,FilledPolygon,Viewer,make_circle,Compound
import matplotlib.pyplot as plt
import numpy as np
import os

# Primitives are 1.0x1.0,
# objects should be composed as 1.0x1.0
# tuples of default_objects are (name,relative scale, list of primitives)
# list tuples are ('BTC', x,y,(scalex,scaley),rot-degrees)
default_objects=[
    ('boat',1,[('B',0,-.43,(1,.15),0),('T',0.1,0.1,(.5,0.85),0)]),
    ('house',1,[('B',0,-.202,(1,.55),0),('T',-.25,0.32,(.5,.4),0),('T',0.25,0.32,(-.5,.4),0)]),
    ('car', 1, [('B', 0, 0, (1, .45), 0),('B', 0,.3, (0.4, .4), 0), ('C', -.25,- 0.32, (.2, .2), 0), ('C', 0.25, -0.32, (.2, .2), 0)]),
]
'''
'''
def generator(width_height=(28, 28), object_scale=0.5,
              width_shift_range=0.25, height_shift_range=0.25,
              scale_range=1, rotate_range=0,
              count=1,
              objects = default_objects):
    """Generate images and labels in a keras data generator style. Image contains simple objects
    in a random pose.

    Args:
        width_height (tuple(width,height)     : The size of the generated images.
        object_scale (float)        : The size objects contained in the images as fraction of image width
        width_shift_range(float)    : The range of random shift in object x position as fraction of image width
        height_shift_range(float)   : The range of random shift in object y position as fraction of image height
        scale_range(float)          : The range of scales of the objects. Objects are scaled (from x to 1)
        rotate_range(float)         : The range of rotation in degrees
        objects (list)              : A description of objects that can be contained in the images.

    Returns:
        generator                   :  The keras data generator. Firt argument is the image. Second is
                                       the pose (class,x offset,y offset, scale, rotation)
    """
    viewer = Viewer(*width_height)
    while 1:
        viewer.geoms=[]
        y_truth=[]
        for i in range(count):
            cls=random.randrange(len(objects))
            obj=[]
            for g,x,y,s,r in objects[cls][2]:
                r*=(np.pi/180)
                if g in 'B':
                    geom = FilledPolygon([(-0.5,-0.5), (0.5,-0.5), (0.5,0.5), (-0.5,0.5)])
                elif g in 'T':
                    geom = FilledPolygon(([(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5)]))
                elif g in 'C':
                    geom=make_circle(radius=1, res=30)
                geom.add_attr(Transform(translation=(x, y), rotation=r, scale=s))
                geom.add_attr(Transform(scale=(objects[cls][1],objects[cls][1])))
                geom.set_color(.8,.6,.4)
                obj.append(geom)

            x= random.uniform(-width_shift_range,width_shift_range)
            y= random.uniform(-height_shift_range,height_shift_range)
            s= random.uniform(scale_range,1)
            r=random.uniform(-rotate_range,rotate_range)*(np.pi/180)
            ss=s*object_scale * width_height[0]

            geom= Compound(obj)
            geom.add_attr(Transform(translation=((x + 0.5)*width_height[0], (y+0.5)*width_height[1]),
                                    rotation=r,scale=(ss,ss)))
            viewer.add_geom(geom)
            y_truth.append((cls,x,y,s,r))

        img=viewer.render(return_rgb_array = True)
        yield (img, np.array(y_truth))



def load_mnist(path):
    # the data, shuffled and split between train and test sets
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    train_rows=y_train.shape[0]
    y_train=np.concatenate([np.reshape(y_train, [train_rows, 1, 1]),np.zeros([train_rows,1,4])],axis=-1)
    test_rows=y_test.shape[0]
    y_test=np.concatenate([np.reshape(y_test, [test_rows, 1, 1]),np.zeros([test_rows,1,4])],axis=-1)
    np.savez_compressed(path, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)


if __name__ == "__main__":
    import argparse

    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="Generate a cache of Synthetic images.")

    parser.add_argument('--count', default=1,type=int,
                        help="Number of object per image")
    parser.add_argument('--train', default=10000,type=int,
                        help="Number of training images")
    parser.add_argument('--test', default=10000,type=int,
                        help="Number of test images")
    parser.add_argument('--file', default="images",
                        help="filename to save training and test data")
    parser.add_argument('--scale', default=0.5,type=float,
                        help="object scale [0.0-1.0]")
    parser.add_argument('--width', default=28,type=int,
                        help="image width in pixels")
    parser.add_argument('--height', default=28,type=int,
                        help="image height in pixels")
    parser.add_argument('--width_shift_range', default=0.25,type=float,
                        help="object shift range [0.0-1.0")
    parser.add_argument('--height_shift_range', default=0.25,type=float,
                        help="object shift range [0.0-1.0")
    parser.add_argument('--scale_range', default=1.0,type=float,
                        help="object scale range [0.0-1.0] (scale down only)")
    parser.add_argument('--rotate_range', default=0.0,type=float,
                        help="object rotate range [0.0-1.0]")
    parser.add_argument('--save_dir', default="./data/",
                        help="object rotate range [0.0-1.0]")
    parser.add_argument('--mnist',  action='store_true',
                        help="object rotate range [0.0-1.0]")

    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    filename=os.path.join(args.save_dir,args.file)

    if args.mnist:
        load_mnist(filename)
        exit()

    gen = generator(width_height=(args.width,args.height), object_scale=args.scale,
              width_shift_range=args.width_shift_range, height_shift_range=args.height_shift_range,
              scale_range=args.scale_range, rotate_range=args.rotate_range,
              count=args.count)

    x, y = [], []
    print("Generating images")
    for i in range(args.train + args.test):
        img, poses = next(gen)
        x.append(img[:, :, 0:1])
        y.append(poses)
    y = np.array(y)
    n_class = int(np.max(y[:, :, 0])) + 1

    x_train = np.array(x[:args.train]).astype('float32') / 255
    y_train = y[:args.train].astype('float32')

    x_test = np.array(x[args.train:]).astype('float32') / 255
    y_test = y[args.train:].astype('float32')

    print("saving images")
    np.savez_compressed(filename, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)