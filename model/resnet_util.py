# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities of ResNet model builder."""

import tensorflow as tf

BN_MOM = 0.9
BN_EPS = 1e-05
NN_AXIS = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1


def convnxn(x,
            filters=64,
            kernel_size=3,
            strides=1,
            use_bias=False,
            kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=2.0, mode='fan_out'),
            name=None):
  """Conv NxN with SAME padding."""
  pad_size = kernel_size // 2
  if pad_size > 0:
    x = tf.keras.layers.ZeroPadding2D(
        ((pad_size, pad_size), (pad_size, pad_size)), name=name + '_pad')(
            x)
  return tf.keras.layers.Conv2D(
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      use_bias=use_bias,
      kernel_initializer=kernel_initializer,
      name=name)(
          x)


def conv3x3(x,
            filters=64,
            strides=1,
            use_bias=False,
            kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=2.0, mode='fan_out'),
            name=None):
  return convnxn(x, filters, 3, strides, use_bias, kernel_initializer, name)


def conv1x1(x,
            filters=64,
            strides=1,
            use_bias=False,
            kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=2.0, mode='fan_out'),
            name=None):
  return convnxn(x, filters, 1, strides, use_bias, kernel_initializer, name)


def normalization_fn(x,
                     normalization='bn',
                     norm_axis=NN_AXIS,
                     bn_mom=BN_MOM,
                     bn_eps=BN_EPS,
                     name=None):
  if normalization == 'bn':
    return tf.keras.layers.BatchNormalization(
        axis=norm_axis, momentum=bn_mom, epsilon=bn_eps, name=name)(
            x)


def nonlinearity(x, activation='relu', name=None):
  """Nonlinearity layers."""
  if activation in ['relu', 'sigmoid']:
    return tf.keras.layers.Activation(activation, name=name)(x)
  elif activation == 'leaky_relu':
    return tf.keras.layers.LeakyReLU(alpha=0.1, name=name)(x)
  else:
    return x


def get_head(x, num_class=2, classifier_activation='linear'):
  """Gets classification head."""
  logits = tf.keras.layers.Dense(
      units=num_class, activation=classifier_activation, name='predictions')(
          x)
  return {'logits': logits}


# pylint: disable=invalid-name
def ResNet(stack_fn,
           preact,
           model_name='resnet',
           input_shape=None,
           pooling=None,
           normalization='bn',
           activation='relu',
           num_class=1000,
           **kwargs):
  """Instantiates the ResNet, ResNetV2, and ResNeXt architecture.

  Reference paper:
  - [Deep Residual Learning for Image Recognition]
      (https://arxiv.org/abs/1512.03385) (CVPR 2015)

  Caution: Be sure to properly pre-process your inputs to the application.
  Please see `applications.resnet.preprocess_input` for an example.

  Args:
    stack_fn: a function that returns output tensor for the stacked residual
      blocks.
    preact: whether to use pre-activation or not (True for ResNetV2, False for
      ResNet and ResNeXt).
    model_name: string, model name.
    input_shape: optional shape tuple.
    pooling: optional pooling mode for feature extraction.
    normalization: batch normalization.
    activation: nonlinear activation type.
    num_class: optional number of classes to classify images into, only to be
      specified if `head` is True, and if no `weights` argument is specified.
    **kwargs: For backwards compatibility only.

  Returns:
    A `keras.Model` instance.

  Raises:
    ValueError: in case of invalid argument for `weights`, or invalid input
    shape.
  """
  del kwargs
  # input layer.
  inputs = img_input = tf.keras.layers.Input(shape=(None, None, input_shape[2]))

  # first block.
  if input_shape[0] in [128, 256]:
    kernel_size, stride, maxpool = 7, 2, True
  elif input_shape[0] in [64, 96]:
    kernel_size, stride, maxpool = 5, 1, True
  elif input_shape[0] == 32:
    kernel_size, stride, maxpool = 3, 1, False
  else:
    raise NotImplementedError
  x = convnxn(
      img_input,
      filters=64,
      kernel_size=kernel_size,
      strides=stride,
      use_bias=preact,
      name='conv1_conv')
  if not preact:
    x = normalization_fn(
        x, normalization=normalization, name='conv1_' + normalization)
    x = nonlinearity(x, activation=activation, name='conv1_' + activation)
  if maxpool:
    x = tf.keras.layers.ZeroPadding2D(((1, 1), (1, 1)), name='pool1_pad')(x)
    x = tf.keras.layers.MaxPooling2D(3, strides=2, name='pool1_pool')(x)

  # residual blocks.
  x = stack_fn(x)
  if preact:
    x = normalization_fn(
        x, normalization=normalization, name='post_' + normalization)
    x = nonlinearity(x, activation=activation, name='post_' + activation)

  # pooling layer
  if pooling in ['avg', None]:
    x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
  elif pooling == 'max':
    x = tf.keras.layers.GlobalMaxPooling2D(name='max_pool')(x)

  # classifier head.
  outputs = get_head(x, num_class=num_class, classifier_activation='linear')

  # create model.
  if pooling not in ['avg', None]:
    model_name += '_{}'.format(pooling)
  if num_class > 0:
    model_name += '_cls{}'.format(num_class)
  return tf.keras.models.Model(inputs, outputs, name=model_name)


def WideResNet(stack_fn,
               preact=True,
               model_name='resnet',
               input_shape=None,
               pooling=None,
               normalization='bn',
               activation='leaky_relu',
               num_class=1000,
               **kwargs):
  """Instantiates the WideResNet architecture.

  Args:
    stack_fn: a function that returns output tensor for the stacked residual
      blocks.
    preact: whether to use pre-activation or not (True for ResNetV2, False for
      ResNet and ResNeXt).
    model_name: string, model name.
    input_shape: optional shape tuple.
    pooling: optional pooling mode for feature extraction.
    normalization: batch normalization.
    activation: nonlinear activation type.
    num_class: optional number of classes to classify images into, only to be
      specified if `head` is True, and if no `weights` argument is specified.
    **kwargs: For backwards compatibility only.

  Returns:
    A `keras.Model` instance.

  Raises:
    ValueError: in case of invalid argument for `weights`, or invalid input
    shape.
  """
  del kwargs

  # input layer.
  inputs = img_input = tf.keras.layers.Input(shape=(None, None, input_shape[2]))

  # first block.
  if input_shape[0] == 256:
    kernel_size, stride, maxpool = 7, 2, True
  elif input_shape[0] in [128]:
    kernel_size, stride, maxpool = 5, 1, True
  elif input_shape[0] in [32, 64, 96]:
    kernel_size, stride, maxpool = 3, 1, False
  else:
    raise NotImplementedError
  x = convnxn(
      img_input,
      filters=16,
      kernel_size=kernel_size,
      strides=stride,
      use_bias=False,
      name='conv1_conv')
  if not preact:
    if normalization == 'bn':
      x = normalization_fn(
          x,
          normalization=normalization,
          bn_mom=0.999,
          bn_eps=0.001,
          name='conv1_bn')
    x = nonlinearity(x, activation=activation, name='conv1_' + activation)
  if maxpool:
    x = tf.keras.layers.ZeroPadding2D(((1, 1), (1, 1)), name='pool1_pad')(x)
    x = tf.keras.layers.MaxPooling2D(3, strides=2, name='pool1_pool')(x)

  # residual blocks.
  x = stack_fn(x)
  if preact:
    if normalization == 'bn':
      x = normalization_fn(
          x,
          normalization=normalization,
          bn_mom=0.999,
          bn_eps=0.001,
          name='post_bn')
    x = nonlinearity(x, activation=activation, name='post_' + activation)

  # pooling layer.
  if pooling in ['avg', None]:
    x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
  elif pooling == 'max':
    x = tf.keras.layers.GlobalMaxPooling2D(name='max_pool')(x)

  # classifier head.
  outputs = get_head(x, num_class=num_class, classifier_activation='linear')

  # create model.
  if num_class > 0:
    model_name += '_cls{}'.format(num_class)
  return tf.keras.models.Model(inputs, outputs, name=model_name)


def block1(x,
           filters,
           bottleneck=False,
           stride=1,
           expansion=1,
           normalization='bn',
           activation='relu',
           name=None):
  """A basic residual block.

  Args:
    x: input tensor.
    filters: integer, filters of the bottleneck layer.
    bottleneck: boolean, if True, use bottlenect block.
    stride: default 1, stride of the first layer.
    expansion: integer, expansion for residual block.
    normalization: string, normalization type.
    activation: string, activation type.
    name: string, block label.

  Returns:
    Output tensor for the residual block.
  """
  conv_shortcut = (stride != 1) or (expansion * filters != x.shape[3])
  if conv_shortcut:
    shortcut = conv1x1(
        x, filters=expansion * filters, strides=stride, name=name + '_0_conv')
    shortcut = normalization_fn(
        shortcut,
        normalization=normalization,
        name=name + '_0_' + normalization)
  else:
    shortcut = x
  # first conv.
  if bottleneck:
    x = conv1x1(x, filters=filters, strides=1, name=name + '_1_conv')
    x = normalization_fn(
        x, normalization=normalization, name=name + '_1_' + normalization)
    x = nonlinearity(x, activation=activation, name=name + '_1_' + activation)
  # second conv.
  idx = 2 if bottleneck else 1
  x = conv3x3(x, filters=filters, strides=stride, name=name + '_%d_conv' % idx)
  x = normalization_fn(
      x,
      normalization=normalization,
      name=name + '_%d_%s' % (idx, normalization))
  x = nonlinearity(
      x, activation=activation, name=name + '_%d_%s' % (idx, activation))
  # last conv.
  last_conv = conv1x1 if bottleneck else conv3x3
  x = last_conv(
      x,
      filters=expansion * filters,
      strides=1,
      name=name + '_%d_conv' % (idx + 1))
  x = normalization_fn(
      x,
      normalization=normalization,
      name=name + '_%d_%s' % (idx + 1, normalization))
  # skip connection.
  x = tf.keras.layers.Add(name=name + '_add')([shortcut, x])
  x = nonlinearity(x, activation=activation, name=name + '_out_' + activation)
  return x


def block2(x,
           filters,
           bottleneck=False,
           stride=1,
           expansion=4,
           normalization='bn',
           activation='relu',
           conv_shortcut=False,
           name=None):
  """A residual block.

  Args:
    x: input tensor.
    filters: integer, filters of the bottleneck layer.
    bottleneck: boolean, if True, use bottlenect block.
    stride: default 1, stride of the first layer.
    expansion: integer, expansion for residual block.
    normalization: string, normalization type.
    activation: string, activation type.
    conv_shortcut: default False, use convolution shortcut if True, otherwise
      identity shortcut.
    name: string, block label.

  Returns:
    Output tensor for the residual block.
  """
  x = normalization_fn(
      x, normalization=normalization, name=name + '_preact_' + normalization)
  x = nonlinearity(
      x, activation=activation, name=name + '_preact_' + activation)
  if conv_shortcut:
    shortcut = conv1x1(
        x, filters=expansion * filters, strides=stride, name=name + '_0_conv')
  else:
    shortcut = tf.keras.layers.MaxPooling2D(
        1, strides=stride)(x) if stride > 1 else x
  # first conv.
  if bottleneck:
    x = conv1x1(x, filters=filters, strides=1, name=name + '_1_conv')
    x = normalization_fn(
        x, normalization=normalization, name=name + '_1_' + normalization)
    x = nonlinearity(x, activation=activation, name=name + '_1_' + activation)
  # second conv.
  idx = 2 if bottleneck else 1
  x = conv3x3(x, filters=filters, strides=stride, name=name + '_%d_conv' % idx)
  x = normalization_fn(
      x,
      normalization=normalization,
      name=name + '_%d_%s' % (idx, normalization))
  x = nonlinearity(
      x, activation=activation, name=name + '_%d_%s' % (idx, activation))
  # last conv
  last_conv = conv1x1 if bottleneck else conv3x3
  x = last_conv(
      x,
      filters=expansion * filters,
      strides=1,
      name=name + '_%d_conv' % (idx + 1))
  # skip connection.
  x = tf.keras.layers.Add(name=name + '_add')([shortcut, x])
  return x


def block3(x,
           filters,
           bottleneck=False,
           stride=1,
           expansion=1,
           normalization='bn',
           activation='leaky_relu',
           conv_shortcut=False,
           use_bias=False,
           name=None):
  """A residual block for WideResNet.

  Args:
    x: input tensor.
    filters: integer, filters of the bottleneck layer.
    bottleneck: boolean, if True, use bottlenect block.
    stride: default 1, stride of the first layer.
    expansion: integer, expansion for residual block.
    normalization: string, normalization type.
    activation: string, activation type.
    conv_shortcut: default False, use convolution shortcut if True, otherwise
      identity shortcut.
    use_bias: boolean, use bias if True.
    name: string, block label.

  Returns:
    Output tensor for the residual block.
  """
  preact = normalization_fn(
      x,
      normalization=normalization,
      bn_mom=0.999,
      bn_eps=0.001,
      name=name + '_preact_' + normalization)
  preact = nonlinearity(
      preact, activation=activation, name=name + '_preact_' + activation)
  shortcut = preact if conv_shortcut else x
  if shortcut.shape[3] != expansion * filters:
    shortcut = conv1x1(
        shortcut,
        filters=expansion * filters,
        strides=stride,
        use_bias=use_bias,
        name=name + '_0_conv')
  # first conv.
  if bottleneck:
    x = conv1x1(
        preact,
        filters=filters,
        strides=1,
        use_bias=use_bias,
        name=name + '_1_conv')
    x = normalization_fn(
        x,
        normalization=normalization,
        bn_mom=0.999,
        bn_eps=0.001,
        name=name + '_1_' + normalization)
    x = nonlinearity(x, activation=activation, name=name + '_1_' + activation)
  # second conv.
  idx = 2 if bottleneck else 1
  x = conv3x3(
      x if bottleneck else preact,
      filters=filters,
      strides=stride,
      use_bias=use_bias,
      name=name + '_%d_conv' % idx)
  x = normalization_fn(
      x,
      normalization=normalization,
      bn_mom=0.999,
      bn_eps=0.001,
      name=name + '_%d_%s' % (idx, normalization))
  x = nonlinearity(
      x, activation=activation, name=name + '_%d_%s' % (idx, activation))
  # last conv.
  last_conv = conv1x1 if bottleneck else conv3x3
  x = last_conv(
      x,
      filters=expansion * filters,
      strides=1,
      use_bias=use_bias,
      name=name + '_%d_conv' % (idx + 1))
  # skip connection.
  x = tf.keras.layers.Add(name=name + '_add')([shortcut, x])
  return x


def stack_v1(x,
             filters,
             blocks,
             bottleneck=False,
             stride1=2,
             expansion=4,
             normalization='bn',
             activation='relu',
             name=None):
  """A stack of residual blocks with block1.

  Args:
    x: input tensor.
    filters: integer, filters of the bottleneck layer in a block.
    blocks: integer, blocks in the stacked blocks.
    bottleneck: boolean, if True, use bottlenect block.
    stride1: default 2, stride of the first layer in the first block.
    expansion: integer, expansion for residual block.
    normalization: string, normalization type.
    activation: string, activation type.
    name: string, stack label.

  Returns:
      Output tensor for the stacked blocks.
  """
  x = block1(
      x,
      filters,
      bottleneck=bottleneck,
      stride=stride1,
      expansion=expansion,
      normalization=normalization,
      activation=activation,
      name=name + '_block1')
  for i in range(1, blocks):
    x = block1(
        x,
        filters,
        bottleneck=bottleneck,
        expansion=expansion,
        normalization=normalization,
        activation=activation,
        name=name + '_block' + str(i + 1))
  return x


def stack_v2(x,
             filters,
             blocks,
             bottleneck=False,
             stride1=2,
             expansion=4,
             normalization='bn',
             activation='relu',
             name=None):
  """A stack of residual blocks with block2.

  Args:
    x: input tensor.
    filters: integer, filters of the bottleneck layer in a block.
    blocks: integer, blocks in the stacked blocks.
    bottleneck: boolean, if True, use bottlenect block.
    stride1: default 2, stride of the first layer in the first block.
    expansion: integer, expansion for residual block.
    normalization: string, normalization type.
    activation: string, activation type.
    name: string, stack label.

  Returns:
      Output tensor for the stacked blocks.
  """
  if blocks == 1:
    x = block2(
        x,
        filters,
        bottleneck=bottleneck,
        stride=stride1,
        expansion=expansion,
        normalization=normalization,
        activation=activation,
        conv_shortcut=True and (stride1 != 1),
        name=name + '_block1')
    return x
  else:
    x = block2(
        x,
        filters,
        bottleneck=bottleneck,
        expansion=expansion,
        normalization=normalization,
        activation=activation,
        conv_shortcut=True and (stride1 != 1),
        name=name + '_block1')
    for i in range(1, blocks - 1):
      x = block2(
          x,
          filters,
          bottleneck=bottleneck,
          expansion=expansion,
          normalization=normalization,
          activation=activation,
          name=name + '_block' + str(i + 1))
    x = block2(
        x,
        filters,
        bottleneck=bottleneck,
        stride=stride1,
        expansion=expansion,
        normalization=normalization,
        activation=activation,
        name=name + '_block' + str(blocks))
    return x


def stack_v3(x,
             filters,
             blocks,
             bottleneck=False,
             stride1=2,
             expansion=4,
             normalization='bn',
             activation='leaky_relu',
             name=None):
  """A stack of residual blocks with block3 (WideResNet).

  Args:
    x: input tensor.
    filters: integer, filters of the bottleneck layer in a block.
    blocks: integer, blocks in the stacked blocks.
    bottleneck: boolean, if True, use bottlenect block.
    stride1: default 2, stride of the first layer in the first block.
    expansion: integer, expansion for residual block.
    normalization: string, normalization type.
    activation: string, activation type.
    name: string, stack label.

  Returns:
      Output tensor for the stacked blocks.
  """
  if blocks == 1:
    x = block3(
        x,
        filters,
        bottleneck=bottleneck,
        stride=stride1,
        expansion=expansion,
        normalization=normalization,
        activation=activation,
        conv_shortcut=True,
        name=name + '_block1')
    return x
  else:
    x = block3(
        x,
        filters,
        bottleneck=bottleneck,
        stride=stride1,
        expansion=expansion,
        normalization=normalization,
        activation=activation,
        conv_shortcut=True,
        name=name + '_block1')
    for i in range(1, blocks):
      x = block3(
          x,
          filters,
          bottleneck=bottleneck,
          expansion=expansion,
          normalization=normalization,
          activation=activation,
          name=name + '_block' + str(i + 1))
    return x


def basic_stack1(x, filters, blocks, **kwargs):
  return stack_v1(x, filters, blocks, bottleneck=False, **kwargs)


def bottleneck_stack1(x, filters, blocks, **kwargs):
  return stack_v1(x, filters, blocks, bottleneck=True, **kwargs)


def basic_stack2(x, filters, blocks, **kwargs):
  return stack_v2(x, filters, blocks, bottleneck=False, **kwargs)


def bottleneck_stack2(x, filters, blocks, **kwargs):
  return stack_v2(x, filters, blocks, bottleneck=True, **kwargs)


def basic_stack3(x, filters, blocks, **kwargs):
  return stack_v3(x, filters, blocks, bottleneck=False, **kwargs)


def bottleneck_stack3(x, filters, blocks, **kwargs):
  return stack_v3(x, filters, blocks, bottleneck=True, **kwargs)
