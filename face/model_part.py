from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.training import moving_averages

TOWER_NAME = 'tower'
UPDATE_OPS_COLLECTION = '_update_ops_'


def _variable_with_weight_decay(name, shape, stddev, wd, trainable=True):
    var = _variable_on_gpu(name, shape, tf.truncated_normal_initializer(stddev=stddev), trainable=trainable)
    if wd:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def _variable_on_gpu(name, shape, initializer, trainable=True):
    var = tf.get_variable(name, shape, initializer=initializer, trainable=trainable)
    return var


def _phase_shift(inputs, r):
    bsize, a, b, c = inputs.get_shape().as_list()
    bsize = tf.shape(inputs)[0] # Handling Dimension(None) type for undefined batch dim
    X = tf.reshape(inputs, (bsize, a, b, r, r))
    X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
    X = tf.split(X, a, 1)  # a, [bsize, b, r, r]
    X = tf.concat([tf.squeeze(x, axis=1) for x in X], 2)  # bsize, b, a*r, r
    X = tf.split(X, b, 1)  # b, [bsize, a*r, r]
    X = tf.concat([tf.squeeze(x, axis=1) for x in X], 2)  # bsize, a*r, b*r
    return tf.reshape(X, (bsize, a*r, b*r, 1))


def subpixel_conv(X, r, output_c=3, color=True):
    '''
    subpixel convolution layer
    Args:
        X: inputs data
        r: magnification ratio
        color: <rgb:True, gray:False>

    Returns: magnificated images

    '''
    print("sub")
    print(X.get_shape())
    if color:
        Xc = tf.split(X, output_c, 3)
        X = tf.concat([_phase_shift(x, r) for x in Xc], 3)
    else:
        X = _phase_shift(X, r)
    return X


def linear_project(scope_name, inputs, output_size, stddev=0.02, bias_start=0.0, reuse=False, with_w=False):
    shape = inputs.get_shape().as_list()
    with tf.variable_scope(scope_name or "Linear") as scope:
        if reuse is True:
            scope.reuse_variables()
        matrix = tf.get_variable("weights", [shape[1], output_size], tf.float32, tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(bias_start))
    if with_w:
        return tf.matmul(inputs, matrix) + bias, matrix, bias
    else:
        return tf.matmul(inputs, matrix) + bias


def residual_block_wout_bn(scope_name, inputs, output_channel, first_block=False, wd=0.0, trainable=True):
    '''
    Defines a residual block in ResNet
    :param inputs: 4D tensor with mini-batch
    :param output_channel: int. return_tensor.get_shape().as_list()[-1] = output_channel
    :param first_block: if this is the first residual block of the whole network
    :return: 4D tensor.
    '''
    input_channel = inputs.get_shape().as_list()[-1]

    # When it's time to "shrink" the image size, we use stride = 2
    if input_channel * 2 == output_channel:
        increase_dim = True
        stride = 2
    elif input_channel == output_channel:
        increase_dim = False
        stride = 1
    else:
        raise ValueError('Output and input channel does not match in residual blocks!!!')

    # The first conv layer of the first residual block does not need to be normalized and relu-ed.
    with tf.variable_scope('conv1_in_block_%s' % (scope_name)):
        if first_block:
            conv1 = conv2d(
                scope_name,
                inputs,
                shape=[3, 3, input_channel, output_channel],
                bias_shape=[output_channel],
                stride=[1, 1, 1, 1],
                padding='SAME')
        else:
            conv1 = lrelu_conv2d(
                scope_name,
                inputs,
                shape=[3, 3, input_channel, output_channel],
                bias_shape=[output_channel],
                stride=[1, 1, 1, 1],
                padding='SAME')

    with tf.variable_scope('conv2_in_block_%s' % (scope_name)):
        conv2 = lrelu_conv2d(scope_name, conv1, shape=[3, 3, input_channel, output_channel], bias_shape=[output_channel], stride=[1, 1, 1, 1], padding='SAME')

    # When the channels of input layer and conv2 does not match, we add zero pads to increase the
    #  depth of input layers
    if increase_dim is True:
        pooled_input = tf.nn.avg_pool(inputs, ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1], padding='VALID')
        padded_input = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [input_channel // 2,
                                                                     input_channel // 2]])
    else:
        padded_input = inputs

    output = conv2 + padded_input
    return output


def residual_block(scope_name, inputs, output_channel, first_block=False, wd=0.0, trainable=True):
    '''
    Defines a residual block in ResNet
    :param inputs: 4D tensor with mini-batch
    :param output_channel: int. return_tensor.get_shape().as_list()[-1] = output_channel
    :param first_block: if this is the first residual block of the whole network
    :return: 4D tensor.
    '''
    input_channel = inputs.get_shape().as_list()[-1]

    # When it's time to "shrink" the image size, we use stride = 2
    if input_channel * 2 == output_channel:
        increase_dim = True
        stride = 2
    elif input_channel == output_channel:
        increase_dim = False
        stride = 1
    else:
        raise ValueError('Output and input channel does not match in residual blocks!!!')

    # The first conv layer of the first residual block does not need to be normalized and relu-ed.
    with tf.variable_scope('conv1_in_block_%s' % (scope_name)):
        if first_block:
            conv1 = conv2d(
                scope_name,
                inputs,
                shape=[3, 3, input_channel, output_channel],
                bias_shape=[output_channel],
                stride=[1, 1, 1, 1],
                padding='SAME')
        else:
            conv1 = bn_relu_conv2d(
                scope_name,
                inputs,
                shape=[3, 3, input_channel, output_channel],
                bias_shape=[output_channel],
                stride=[1, 1, 1, 1],
                padding='SAME')

    with tf.variable_scope('conv2_in_block_%s' % (scope_name)):
        conv2 = bn_relu_conv2d(scope_name, conv1, shape=[3, 3, input_channel, output_channel], bias_shape=[output_channel], stride=[1, 1, 1, 1], padding='SAME')

    # When the channels of input layer and conv2 does not match, we add zero pads to increase the
    #  depth of input layers
    if increase_dim is True:
        pooled_input = tf.nn.avg_pool(inputs, ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1], padding='VALID')
        padded_input = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [input_channel // 2,
                                                                     input_channel // 2]])
    else:
        padded_input = inputs

    output = conv2 + padded_input
    return output


def conv2d(scope_name, inputs, shape, bias_shape, stride, padding='VALID', wd=0.0, reuse=False, trainable=True, with_w=False):
    with tf.variable_scope(scope_name) as scope:
        if reuse is True:
            scope.reuse_variables()
        kernel = _variable_with_weight_decay(
            'weights',
            shape=shape,
            stddev=0.02,
            wd=wd,
            trainable=trainable
        )
        print("inputs")
        print(inputs.get_shape())
        conv = tf.nn.conv2d(inputs, kernel, stride, padding=padding)
        biases = _variable_on_gpu('biases', bias_shape, tf.constant_initializer(0.0), trainable=trainable)
        #conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        print("conv2d")
        print(conv.get_shape())
        conv = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))
        print(conv.get_shape())

        # bias = tf.nn.bias_add(conv, biases)
        # conv_ = tf.nn.relu(bias, name=scope.name)
    if with_w:
        print(conv.get_shape())
        return conv, kernel, biases
    else:
        return conv


def lrelu_conv2d(scope_name, inputs, shape, bias_shape, stride, padding='VALID', wd=0.0, reuse=False, trainable=True, with_w=False):
    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()

        # relu
        relu = lrelu(inputs)

        # conv2d
        kernel = _variable_with_weight_decay(
            'weights',
            shape=shape,
            stddev=0.01,
            wd=wd,  # not use weight decay
            trainable=trainable
        )
        conv = tf.nn.conv2d(relu, kernel, stride, padding=padding)
        if with_w:
            return conv, kernel
        else:
            return conv


def bn_relu_conv2d(scope_name, inputs, shape, bias_shape, stride, padding='VALID', wd=0.0, reuse=False, trainable=True, with_w=False):
    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()
        # bn layer
        bn = batch_norm(inputs)

        # relu
        relu = tf.nn.relu(bn)

        # conv2d
        kernel = _variable_with_weight_decay(
            'weights',
            shape=shape,
            stddev=0.01,
            wd=wd,  # not use weight decay
            trainable=trainable
        )
        conv = tf.nn.conv2d(relu, kernel, stride, padding=padding)
        if with_w:
            return conv, kernel
        else:
            return conv


def conv2d_bn(scope_name, inputs, shape, bias_shape, stride, padding='VALID', wd=0.0, reuse=False, trainable=True, with_w=False):
    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()
        kernel = _variable_with_weight_decay(
            'weights',
            shape=shape,
            stddev=0.01,
            wd=wd,  # not use weight decay
            trainable=trainable
        )
        conv = tf.nn.conv2d(inputs, kernel, stride, padding=padding)
        bn = batch_norm(conv)
        # conv_ = tf.nn.relu(bn, name=scope.name)
        if with_w:
            return bn, kernel
        else:
            return bn


def conv2d_transpose(scope_name, inputs, shape, output_shape, bias_shape, stride, padding='VALID', reuse=False, trainable=True, with_w=False):
    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()
        kernel = _variable_with_weight_decay(
            'weights',
            shape=shape,
            stddev=0.02,
            wd=0.0,  # not use weight decay
            trainable=trainable
        )
        tf_output_shape = tf.stack(output_shape)
        deconv = tf.nn.conv2d_transpose(inputs, kernel, tf_output_shape, stride, padding=padding)
        deconv.set_shape(output_shape)
        biases = _variable_on_gpu('biases', bias_shape, tf.constant_initializer(0.0), trainable=trainable)
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        #deconv_ = tf.nn.relu(bn, name=scope.name)
        if with_w:
            return deconv, kernel, biases
        else:
            return deconv


def conv2d_transpose_bn(scope_name, inputs, shape, output_shape, bias_shape, stride, padding='VALID', reuse=False, trainable=True, with_w=False):
    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()
        kernel = _variable_with_weight_decay(
            'weights',
            shape=shape,
            stddev=0.01,
            wd=0.0,  # not use weight decay
            trainable=trainable
        )
        tf_output_shape = tf.stack(output_shape)
        deconv = tf.nn.conv2d_transpose(inputs, kernel, tf_output_shape, stride, padding=padding)
        deconv.set_shape(output_shape)
        bn = batch_norm(deconv)
        #deconv_ = tf.nn.relu(bn, name=scope.name)
        if with_w:
            return bn, kernel
        else:
            return bn


class BatchNorm(object):
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, trainable=True):
        return tf.contrib.layers.batch_norm(x,
                                            decay=self.momentum,
                                            updates_collections=None,
                                            epsilon=self.epsilon,
                                            scale=True,
                                            is_training=trainable,
                                            scope=self.name)


def batch_norm(inputs,
                       scope_name,
                       decay=0.999,
                       center=True,
                       scale=False,
                       epsilon=0.001,
                       moving_vars='moving_vars',
                       activation=None,
                       is_training=True,
                       trainable=True,
                       restore=True,
                       scope=None,
                       reuse=None):
    """Adds a Batch Normalization layer.
    Args:
      inputs: a tensor of size [batch_size, height, width, channels]
              or [batch_size, channels].
      decay: decay for the moving average.
      center: If True, subtract beta. If False, beta is not created and ignored.
      scale: If True, multiply by gamma. If False, gamma is
        not used. When the next layer is linear (also e.g. ReLU), this can be
        disabled since the scaling can be done by the next layer.
      epsilon: small float added to variance to avoid dividing by zero.
      moving_vars: collection to store the moving_mean and moving_variance.
      activation: activation function.
      is_training: whether or not the model is in training mode.
      trainable: whether or not the variables should be trainable or not.
      restore: whether or not the variables should be marked for restore.
      scope: Optional scope for variable_op_scope.
      reuse: whether or not the layer and its variables should be reused. To be
        able to reuse the layer scope must be given.
    Returns:
      a tensor representing the output of the operation.
    """
    inputs_shape = inputs.get_shape()
    with tf.variable_scope(scope_name, [inputs], scope, reuse=reuse):
        axis = list(range(len(inputs_shape) - 1))
        params_shape = inputs_shape[-1:]
        # Allocate parameters for the beta and gamma of the normalization.
        beta, gamma = None, None
        if center:
            beta = tf.get_variable('beta',
                                      params_shape,
                                      initializer=tf.zeros_initializer(),
                                      trainable=trainable)
        if scale:
            gamma = tf.get_variable('gamma',
                                       params_shape,
                                       initializer=tf.ones_initializer(),
                                       trainable=trainable)
        moving_collections = [moving_vars, tf.GraphKeys.MOVING_AVERAGE_VARIABLES]
        moving_mean = tf.get_variable('moving_mean',
                                         params_shape,
                                         initializer=tf.zeros_initializer(),
                                         trainable=False)
        moving_variance = tf.get_variable('moving_variance',
                                             params_shape,
                                             initializer=tf.ones_initializer(),
                                             trainable=False)

        if is_training:
            # Calculate the moments based on the individual batch.
            mean, variance = tf.nn.moments(inputs, axis)

            update_moving_mean = moving_averages.assign_moving_average(
                moving_mean, mean, decay)
            tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
            update_moving_variance = moving_averages.assign_moving_average(
                moving_variance, variance, decay)
            tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)
        else:
            # Just use the moving_mean and moving_variance.
            mean = moving_mean
            variance = moving_variance
        # Normalize the activations.
        outputs = tf.nn.batch_normalization(
            inputs, mean, variance, beta, gamma, epsilon)
        outputs.set_shape(inputs.get_shape())
        if activation:
            outputs = activation(outputs)
        return outputs


def fc(scope_name, inputs, shape, bias_shape, wd=0.04, reuse=False, trainable=True):
    with tf.variable_scope(scope_name) as scope:
        if reuse is True:
            scope.reuse_variables()
        # weights = _variable_with_weight_decay(
        #     'weights',
        #     shape,
        #     stddev=0.1,
        #     wd=wd,
        #     trainable=trainable
        # )
        weights = _variable_on_gpu('weights', shape,tf.random_normal_initializer(), trainable=trainable)
        biases = _variable_on_gpu('biases', bias_shape, tf.constant_initializer(0.0), trainable=trainable)
        fc = tf.nn.tanh(tf.add(tf.matmul(inputs, weights), biases, name=scope.name))
        return fc, weights, biases


def fc_relu(scope_name, inputs, shape, bias_shape, wd=0.04, reuse=False, trainable=True):
    with tf.variable_scope(scope_name) as scope:
        if reuse is True:
            scope.reuse_variables()
        # weights = _variable_with_weight_decay(
        #     'weights',
        #     shape,
        #     stddev=0.01,
        #     wd=wd,
        #     trainable=trainable
        # )
        weights = _variable_on_gpu('weights', shape, tf.random_normal_initializer(), trainable=trainable)
        biases = _variable_on_gpu('biases', bias_shape, tf.constant_initializer(0.0), trainable=trainable)
        fc = tf.nn.relu_layer(inputs, weights, biases, name=scope.name)
        return fc, weights, biases


def fc_flat_relu(scope_name, inputs, shape, bias_shape, wd=0.04, reuse=False, trainable=True):
    with tf.variable_scope(scope_name) as scope:
        if reuse is True:
            scope.reuse_variables()
        print(shape)
        flat = tf.reshape(inputs, [-1, shape[0]])
        # weights = _variable_with_weight_decay(
        #     'weights',
        #     shape,
        #     stddev=0.01,
        #     wd=wd,
        #     trainable=trainable
        # )
        weights = _variable_on_gpu('weights', shape, tf.random_normal_initializer(), trainable=trainable)
        biases = _variable_on_gpu('biases', bias_shape, tf.constant_initializer(0.0), trainable=trainable)
        fc = tf.nn.relu_layer(flat, weights, biases, name=scope.name)
        return fc, weights, biases


def fc_softmax(scope_name, inputs, shape, bias_shape, wd=0.04, reuse=False, trainable=True):
    with tf.variable_scope(scope_name) as scope:
        if reuse is True:
            scope.reuse_variables()
        flat = tf.reshape(inputs, [-1, shape[0]])
        weights = _variable_with_weight_decay(
            'weights',
            shape,
            stddev=0.01,
            wd=wd,
            trainable=trainable
        )
        biases = _variable_on_gpu('biases', bias_shape, tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(flat, weights), biases, name=scope.name)
        return softmax_linear, weights, biases


def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * abs(x)