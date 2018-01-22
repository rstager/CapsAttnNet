from keras import layers, initializers
import keras.backend as K
import tensorflow as tf
import numpy as np


dim_geom=6 # Number of dimensions used for the geometric pose

def squash_scale(vectors, axis=-1):
    """
    The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
    :param vectors: some vectors to be squashed, N-dim tensor
    :param axis: the axis to squash
    :return: a Tensor with same shape as input vectors
    """
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
    return scale


class CAN(layers.Layer):
    """
    The capsule attention layer. Similar to a CapsuleLayer, but
    1) shares weights for multiple instances of each capsule type
    2) shares weights for multiple parts of each capsule
    3) uses geometric pose to focus attention

    So its input shape = [None, input_num_capsule, input_num_instance,input_dim_capsule] and output shape = \
    [None, num_capsule, dim_capsule]. For Dense Layer, input_dim_capsule = dim_capsule = 1.

    :param num_capsule: number of capsules types in this layer
    :param dim_capsule_attr: dimension of the output vectors of the capsules (not including geometric pose)
    :param num_instance: number of instances of each capsules type
    :param num_part: number of lower level parts that can compose a capsules
    :param routings: number of iterations for the routing algorithm
    """

    def __init__(self, num_capsule, dim_capsule_attr, num_instance=5,num_part=7, routings=3,
                 kernel_initializer='glorot_uniform',
                 **kwargs):
        super(CAN, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.num_instance = num_instance
        self.num_part = num_part
        self.dim_capsule = dim_capsule_attr+dim_geom+1
        self.dim_attr = dim_capsule_attr
        self.routings = routings
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        assert len(input_shape) >= 3, "The input Tensor should have shape=[None, input_num_capsule, input_num_instance, input_dim_capsule]"
        self.input_num_capsule = input_shape[1]
        self.input_num_instance = input_shape[2]
        self.input_dim_capsule = input_shape[3]
        self.input_dim_attr = self.input_dim_capsule - dim_geom - 1

        # Transform matrix for geometric pose
        self.W1 = self.add_weight(shape=[self.num_capsule, self.num_part,
                                         dim_geom+1, dim_geom],
                                 initializer=self.kernel_initializer,
                                 name='W1')

        # Tranform matrix for attributes
        self.W2 = self.add_weight(shape=[self.num_capsule,self.num_part,
                                         self.input_dim_attr,self.dim_attr],
                                 initializer=self.kernel_initializer,
                                 name='W2')

        self.built = True

    def _part_to_whole_predictions(self, x):
        """
        Estimate the pose of the whole given the pose of the part.
        :param x: set of poses to transform
        """
        # inputs.shape=[ input_num_capsule, input_num_instance, input_dim_capsule]
        # output.shape=[num_instance*num_capsule,num_parts*input_num_capsule*input_num_instance,dim_capsule]
        # xt.shape = [ num_instance, input_num_capsule, input_num_instance, num_capsule, num_part, input_dim_capsule]
        # xpart.shape = [ num_instance, input_num_capsule, input_num_instance, num_capsule, num_part, input_dim_capsule]

        xt = K.tile(K.expand_dims(x,0),[self.num_instance,1,1,1])

        ppart = K.reshape( xt[:,:,:,:1],[self.num_instance,self.input_num_capsule,self.input_num_instance,1,1,1])
        ppart = K.tile(ppart,[1,1,1,self.num_capsule,self.num_part,1])

        gpose = K.concatenate([xt[:,:,:,1:dim_geom+1],K.ones_like(xt[:,:,:,:1])]) # add 1 col to allow x-y translate
        gpart = K.dot(gpose, self.W1)
        apart = K.dot(xt[:,:,:,dim_geom+1:], self.W2)
        whole=K.concatenate([ppart,gpart,apart])
        output=K.permute_dimensions(whole,[0,3,4,1,2,5])
        output=K.reshape(output,[self.num_instance*self.num_capsule,
                                 self.num_part*self.input_num_capsule*self.input_num_instance,self.dim_capsule])
        # output = tf.Print(output, [tf.shape(x)], message='x', summarize=16)
        # output = tf.Print(output, [x[0,18,1:3]], message='x ', summarize=3)
        # output = tf.Print(output, [gpose[0,0,0,:]], message='x gpose ', summarize=5)
        # output = tf.Print(output, [gpose[0,1,0,:]], message='x gpose ', summarize=5)
        # output = tf.Print(output, [gpart[0,0,0,0,0,:]], message='x gpart ', summarize=5)
        # output = tf.Print(output, [gpart[0,1,0,0,0,:]], message='x gpart ', summarize=5)
        return output

    def _best_guess(self, c, inputs_hat):
        '''
        Combine the predicted poses 'input_hats' weighted by c to come up with best_guess of the capsule poses

        :param c: weights to apply to the input poses
        :param inputs_hat: input poses
        :return: best guess at pose
        '''
        # c.shape=[None, num_capsule * num_instance, num_part * input_num_capsule * input_num_instance]
        # inputs_hat.shape = [None,num_instance * num_capsule, num_parts, input_num_capsule * input_num_instance, dim_capsule]
        # guess.shape = [None,num_instance * num_capsule,dim_capsule]

        # take the mean probility
        probability = tf.reduce_mean(inputs_hat[:,:,:,0:1],axis=2)

        # find the mean weighted geometric pose
        sum_weighted_geoms = K.batch_dot(c,inputs_hat[:,:,:,1:dim_geom+1], [2, 2])
        one_over_weight_sums = tf.tile(tf.expand_dims(tf.reciprocal(K.sum(c,axis=-1)),-1),[1,1,dim_geom])
        mean_geom =  one_over_weight_sums*sum_weighted_geoms

        # squash the weighted sum of attributes
        weighted_attrs = K.batch_dot(c,inputs_hat[:,:,:,dim_geom+1:], [2, 2])
        scale = squash_scale(weighted_attrs)
        squashed_attrs  = scale*weighted_attrs

        # use the magnitude of the attributes for probability
        probability = scale

        guess = layers.concatenate([probability,mean_geom,squashed_attrs])
        return guess

    def _agreement(self, outputs, inputs_hat):
        '''
        Measure the fit of each predicted poses to the best guess pose and return an adjustment value for the routing
        coefficients

        :param outputs: the best guess estimate of whole pose
        :param inputs_hat: the per part estimate of the whole pose
        :return: adjustment factor to the routing coefficients
        '''

        # outputs.shape = [None, num_instance * num_capsule, dim_capsule]
        # inputs_hat.shape = [None,num_instance * num_capsule, num_parts * input_num_capsule * input_num_instance, dim_capsule]
        # x_agree.shape = [None,num_instance * num_capsule, num_parts*input_num_capsule * input_num_instance],
        # b.shape=[None,num_instance * num_capsule, num_parts*input_num_capsule * input_num_instance]

        geom_agree = K.batch_dot(outputs[:,:,1:dim_geom+1], inputs_hat[:,:,:,1:dim_geom+1], [2, 3])
        attr_agree = K.batch_dot(outputs[:,:,dim_geom+1:], inputs_hat[:,:,:,dim_geom+1:], [2, 3])
        attr_agree *= 0.01

        # geom_agree=tf.Print(geom_agree, [outputs[0,0,:dim_geom+1]], message='agree guess ', summarize=5)
        # geom_agree=tf.Print(geom_agree, [inputs_hat[0,0,0,:dim_geom+1]], message='agree uhat ', summarize=5)
        # geom_agree=tf.Print(geom_agree, [geom_agree[0,0,0]], message='geom_agree ', summarize=5)
        # geom_agree=tf.Print(geom_agree, [attr_agree[0,0,0]], message='attr_agree ', summarize=5)
        # geom_agree=tf.Print(geom_agree, [tf.reduce_max(geom_agree),tf.reduce_min(geom_agree)], message='geom_agree max/min', summarize=5)
        # geom_agree=tf.Print(geom_agree, [tf.reduce_max(attr_agree),tf.reduce_min(attr_agree)], message='attr_agree max/min', summarize=5)

        return geom_agree+attr_agree

    def call(self, inputs, training=None):
        # inputs.shape=[None, input_num_capsule, input_num_instance, input_dim_capsule]
        # inputs_hat.shape=[None,num_instance*num_capsule,num_parts*input_num_capsule*input_num_instance,dim_capsule]

        inputs_hat = K.map_fn(lambda x: self._part_to_whole_predictions(x), elems=inputs)

        # Begin: Routing algorithm ---------------------------------------------------------------------#
        # The prior for coupling coefficient, initialized as zeros.
        # b.shape = [None, self.num_capsule, self.num_parts, self.input_num_capsule].
        b = K.tf.zeros(shape=[K.shape(inputs_hat)[0], self.num_instance*self.num_capsule,
                                 self.num_part*self.input_num_capsule*self.input_num_instance])

        assert self.routings > 0, 'The routings should be > 0.'
        for i in range(self.routings):
            # c.shape=[batch_size, num_instance*num_capsule, input_num_capsule]
            tmpb = K.reshape(b, [-1,self.num_capsule * self.num_instance*self.num_part,
                                        self.input_num_capsule * self.input_num_instance])

            # softmax for all outputs of each input_capsule*input_instance
            tmpc = K.tf.nn.softmax(tmpb, dim=1)
            c=K.reshape(tmpc,[-1,self.num_capsule * self.num_instance,
                                self.num_part*self.input_num_capsule * self.input_num_instance])

            #outputs.shape=[None,num_instance * num_capsule,dim_capsule]
            outputs = self._best_guess(c, inputs_hat)

            if i < self.routings - 1: #
                b += self._agreement(outputs, inputs_hat)

        # End: Routing algorithm -----------------------------------------------------------------------#
        outputs=K.reshape(outputs,[-1,self.num_instance,self.num_capsule,self.dim_capsule])
        outputs=K.permute_dimensions(outputs,[0,2,1,3])
        return outputs

    def compute_output_shape(self, input_shape):
        return tuple([None, self.num_capsule, self.num_instance, self.dim_capsule])

def PrimaryCap(inputs,num_capsule, dim_capsule_attr, kernel_size, strides, padding):
    """
    Apply Conv2D `n_channels` times and concatenate all capsules
    :param inputs: 4D tensor, shape=[None, width, height, channels]
    :param dim_capsule: the dim of the output vector of capsule (not including geometric pose)
    :param n_channels: the number of types of capsules
    :return: output.shape=[None, num_capsule, num_instance, dim_capsule]
    """
    # The pose will contain a probability, a geometric pose data (i.e. location) and attributes.

    dim_capsule=dim_capsule_attr+dim_geom+1

    def build_geom_pose(x):
        '''
        build a PrimaryCap output from the Conv2D layer
        :param x: attributes from input Conv2D
        :return:
        '''
        _,rows,cols,num_capsule,dim_attr = x.shape
        dim_capsule=dim_attr+dim_geom+1
        # create the probability part
        s_squared_norm = K.sum(K.square(x), -1, keepdims=True)
        probability = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())

        # create the xy location part
        bsz=tf.shape(x)[0]

        xcoord, ycoord = tf.meshgrid(tf.linspace(-1.0, 1.0, rows),
                                     tf.linspace(-1.0, 1.0, cols))
        xcoord = tf.reshape(xcoord, [1, rows,cols,1, 1])
        ycoord = tf.reshape(ycoord, [1, rows,cols,1, 1])

        xcoordtiled = tf.tile(xcoord, [bsz,1,1,num_capsule, 1])
        ycoordtiled = tf.tile(ycoord, [bsz,1,1,num_capsule, 1])

        # create the angles + scale part
        # [[cos(a)*scalex,-sin(a)*scaley],[sin(a)*scalex,cos(a)*scaley]]
        a1=tf.ones_like(probability)
        angles=tf.tile(a1,[1,1,1,1,4])

        # now assemble the capsule output
        o1=tf.concat([probability,xcoordtiled, ycoordtiled,angles,x],axis=-1)
        o2=tf.reshape(o1,[bsz,rows*cols,num_capsule,dim_capsule],name="primary_cap_build_pose_output_reshaping")
        out=tf.transpose(o2,[0,2,1,3])
        #out=tf.Print(out,[out[0,0,0,:]],message="primary cap output",summarize=100)
        return out


    output = layers.Conv2D(filters=num_capsule*dim_capsule_attr, kernel_size=kernel_size, strides=strides, padding=padding,
                           name='primarycap_conv2d')(inputs)
    _ , rows, cols, channels = output.shape

    attroutputs = layers.Reshape(target_shape=[int(rows),int(cols),num_capsule,dim_capsule_attr], name='primarycap_attributes')(output)

    outputs=layers.Lambda(build_geom_pose, name='primarycap')(attroutputs)

    return outputs
