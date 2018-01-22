"""
Capsule Attention Network.. a variant of the implementation of CapsNet in Hinton's paper Dynamic Routing Between Capsules.

Derived from work by Xifeng Guo, E-mail: `guoxifeng1990@163.com`, Github: `https://github.com/XifengGuo/CapsNet-Keras`

Usage:
       python train.py
       ... ...

Result:

"""

import numpy as np

import gen_images
from keras.layers import Lambda

from canlayer import PrimaryCap, CAN
from keras import layers, models, optimizers
from keras import backend as K
import tensorflow as tf


K.set_image_data_format('channels_last')


def create_model(input_shape, n_class, n_instance, n_part, routings):
    """
    A Capsule Network on MNIST.
    :param input_shape: data shape, 3d, [width, height, channels]
    :param n_class: number of classes
    :param n_instance: number of instance of each class
    :param n_part: number of parts in each instance
    :param routings: number of routing iterations
    :return: Two Keras Models, the first one used for training, and the second one for evaluation.
            `eval_model` can also be used for training.
    """
    x = layers.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(filters=32, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(x)

    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primarycaps = PrimaryCap(conv1, dim_capsule_attr=2, num_capsule=32, kernel_size=9, strides=2, padding='valid')


    # Layer 3: Capsule layer. Attention algorithm works here.
    digitcaps = CAN(num_capsule=n_class, dim_capsule_attr=10, routings=routings, num_instance=n_instance, num_part=n_part,
                    name='digitcaps')(primarycaps)


    # Layer 4: Convert capsule probabilities to a classification

    out_caps = Lambda(lambda x: x[:, :, :, 0],name='select_probability')(digitcaps)
    out_caps = layers.Permute([2, 1], name='capsnet')(out_caps)  # for clasification we swap order to be instance,class

    # Models for training and evaluation (prediction)
    model = models.Model([x], [out_caps])

    return model  #


def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes, n_instance]
    :param y_pred: [None, n_classes, n_instance]
    :return: a scalar loss value.
    """

    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    loss = K.mean(K.sum(L, 1))

    acc = K.equal(K.argmax(y_true, axis=1), K.argmax(y_pred, axis=1))

    loss = tf.Print(loss,[tf.shape(y_true)],message=" margin loss y_true shape",summarize=6,first_n=1)
    loss = tf.Print(loss,[tf.shape(y_pred)],message=" margin loss y_pred shape",summarize=6,first_n=1)
    loss = tf.Print(loss,[tf.shape(L)],message=" margin loss L shape",summarize=6,first_n=1)
    loss = tf.Print(loss,[tf.shape(acc)],message=" margin loss acc shape",summarize=6,first_n=1)
    # loss = tf.Print(loss,[y_true[0,0,:],y_pred[0,0,:]],message=" margin loss y_true/y_pred",summarize=20)
    # loss = tf.Print(loss,[L[0,0,:]],message=" margin loss L",summarize=6)
    # loss = tf.Print(loss,[loss],message=" margin loss loss",summarize=6)
    # loss = tf.Print(loss,[acc[0,0]],message=" margin loss acc",summarize=6)
    return loss


def train(model, train_gen,test_gen, args):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param data: a tuple containing training and testing data, like `((x_train, y_train), (x_test, y_test))`
    :param args: arguments
    :return: The trained model
    """

    # callbacks
    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
                               batch_size=args.batch_size, histogram_freq=int(args.debug))
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', monitor='val_capsnet_acc',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))

    # compile the model
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=[margin_loss],
                  loss_weights=[1.],
                  metrics={'capsnet': 'accuracy'})

    # Training without data augmentation.
    model.fit_generator(generator=train_gen,
                        steps_per_epoch=args.steps_per_epoch,
                        epochs=args.epochs,
                        validation_data=test_gen,
                        validation_steps=args.validation_steps,
                        callbacks=[log, tb, checkpoint, lr_decay])
    # End: Training with data augmentation -----------------------------------------------------------------------#

    model.save_weights(args.save_dir + '/trained_model.h5')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)
    return model

def table_generator(x,y,bsz=32):
    while True:
        i=0
        for i in range(0,x.shape[0],bsz):
            yield x[i:i+bsz],y[i:i+bsz]

def onehot_generator(generator,dim):
    while True:
        x,y = generator.__next__()
        y_onehot = np.eye(dim)[y.astype('int32')]
        yield (x,y_onehot)

def cached_onehot_generators(data_dir="./data/",filename="images"):
    if not ".npz" in filename:
        filename+=".npz"
    pathname=os.path.join(data_dir,filename)
    try:
        data = np.load(pathname)
        x_test = data['x_test']
        y_test = data['y_test']
        x_train = data['x_train']
        y_train = data['y_train']
    except:
        print("Image cache not found. Use gen_images to generate cached images.")
        exit()


    n_class = int(np.max(y_train)) + 1
    return (onehot_generator(table_generator(x_train,y_train),n_class),
                            onehot_generator(table_generator(x_test,y_test),n_class))

if __name__ == "__main__":
    import os
    import argparse
    from keras.preprocessing.image import ImageDataGenerator
    from keras import callbacks

    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="Capsule Network on MNIST.")
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--steps_per_epoch', default=100, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--validation_steps', default=100, type=int)
    parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.9, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('--lam_recon', default=0.392, type=float,
                        help="The coefficient for the loss of decoder")
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    parser.add_argument('--file', default='images',
                        help="filename for cached images.. see gen_images")
    parser.add_argument('--data', default="./data/",
                        help="data directory for cached images")
    parser.add_argument('--count', default=1,
                        help="Number of object per image")
    parser.add_argument('--npart', default=10,
                        help="Number of parts per object")
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # it might be nice to support non-file generators, but this seems to run faster
    train_gen, test_gen = cached_onehot_generators(args.data,args.file)

    # define model
    x,y=next(train_gen)
    nclass = y.shape[2]
    model = create_model(input_shape=x.shape[1:],
                         n_class=nclass,
                         n_instance=args.count, n_part=args.npart,
                         routings=args.routings)
    model.summary()

    model.compile(optimizer=optimizers.Adam(lr=args.lr), loss=margin_loss, metrics={'capsnet': 'accuracy'})

    # train or test
    if args.weights is not None:  # init the model weights with provided one
        model.load_weights(args.weights)

    train(model, train_gen,test_gen, args=args)

