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
"""ResNet model definition."""

import functools
from model.resnet_util import basic_stack1
from model.resnet_util import basic_stack2
from model.resnet_util import basic_stack3
from model.resnet_util import bottleneck_stack1
from model.resnet_util import bottleneck_stack2
from model.resnet_util import ResNet
from model.resnet_util import WideResNet

__all__ = [
    'ResNet18',
    'ResNet50',
    'ResNet101',
    'ResNet152',
    'ResNet18V2',
    'ResNet50V2',
    'ResNet101V2',
    'ResNet152V2',
    'WRN28',
    'WRN37',
]

BLOCK = {
    'ResNet18': [2, 2, 2, 2],
    'ResNet50': [3, 4, 6, 3],
    'ResNet101': [3, 4, 23, 3],
    'ResNet152': [3, 8, 36, 3],
    'ResNet18v2': [2, 2, 2, 2],
    'ResNet50v2': [3, 4, 6, 3],
    'ResNet101v2': [3, 4, 23, 3],
    'ResNet152v2': [3, 8, 36, 3],
    'WRN28': [4, 4, 4],
    'WRN37': [4, 4, 4, 4],
}

EXPANSION = {
    'ResNet18': 1,
    'ResNet50': 4,
    'ResNet101': 4,
    'ResNet152': 4,
    'ResNet18v2': 1,
    'ResNet50v2': 4,
    'ResNet101v2': 4,
    'ResNet152v2': 4,
    'WRN28': 1,
    'WRN37': 1,
}

STACK = {
    'ResNet18': basic_stack1,
    'ResNet50': bottleneck_stack1,
    'ResNet101': bottleneck_stack1,
    'ResNet152': bottleneck_stack1,
    'ResNet18v2': basic_stack2,
    'ResNet50v2': bottleneck_stack2,
    'ResNet101v2': bottleneck_stack2,
    'ResNet152v2': bottleneck_stack2,
    'WRN28': basic_stack3,
    'WRN37': basic_stack3,
}


def ResNetV1(arch='ResNet18',
             input_shape=None,
             num_class=1000,
             pooling=None,
             normalization='bn',
             activation='relu',
             width=1.0,
             **kwargs):
  """Instantiates the ResNet architecture."""
  del kwargs

  # pylint: disable=invalid-name
  def stack_fn(x, arch, width=1.0):
    block, stack, expansion = BLOCK[arch], STACK[arch], EXPANSION[arch]
    x = stack(
        x,
        int(64 * width),
        block[0],
        expansion=expansion,
        stride1=1,
        normalization=normalization,
        activation=activation,
        name='conv2')
    x = stack(
        x,
        int(128 * width),
        block[1],
        expansion=expansion,
        normalization=normalization,
        activation=activation,
        name='conv3')
    x = stack(
        x,
        int(256 * width),
        block[2],
        expansion=expansion,
        normalization=normalization,
        activation=activation,
        name='conv4')
    return stack(
        x,
        int(512 * width),
        block[3],
        expansion=expansion,
        normalization=normalization,
        activation=activation,
        name='conv5')

  return ResNet(
      stack_fn=functools.partial(stack_fn, arch=arch, width=width),
      preact=False,
      model_name='{}_w{:g}_{}_{}'.format(arch, width, normalization,
                                         activation),
      input_shape=input_shape,
      pooling=pooling,
      normalization=normalization,
      activation=activation,
      num_class=num_class)


def ResNetV2(arch='ResNet18',
             input_shape=None,
             num_class=1000,
             pooling=None,
             normalization='bn',
             activation='relu',
             width=1.0,
             **kwargs):
  """Instantiates the ResNet V2 architecture."""
  del kwargs

  # pylint: disable=invalid-name
  def stack_fn(x, arch, width=1.0):
    block, stack, expansion = BLOCK[arch], STACK[arch], EXPANSION[arch]
    x = stack(
        x,
        int(64 * width),
        block[0],
        expansion=expansion,
        normalization=normalization,
        activation=activation,
        name='conv2')
    x = stack(
        x,
        int(128 * width),
        block[1],
        expansion=expansion,
        normalization=normalization,
        activation=activation,
        name='conv3')
    x = stack(
        x,
        int(256 * width),
        block[2],
        expansion=expansion,
        normalization=normalization,
        activation=activation,
        name='conv4')
    return stack(
        x,
        int(512 * width),
        block[3],
        expansion=expansion,
        stride1=1,
        normalization=normalization,
        activation=activation,
        name='conv5')

  return ResNet(
      stack_fn=functools.partial(stack_fn, arch=arch, width=width),
      preact=True,
      model_name='{}_w{:g}_{}_{}'.format(arch, width, normalization,
                                         activation),
      input_shape=input_shape,
      pooling=pooling,
      normalization=normalization,
      activation=activation,
      num_class=num_class)


def WRN(arch='WRN28',
        input_shape=None,
        num_class=1000,
        pooling=None,
        normalization='bn',
        activation='leaky_relu',
        width=1.0,
        **kwargs):
  """Instantiates the WideResNet architecture."""
  del kwargs

  # pylint: disable=invalid-name
  def stack_fn(x, arch, width=1.0):
    block, stack, expansion = BLOCK[arch], STACK[arch], EXPANSION[arch]
    for i, b in enumerate(block):
      x = stack(
          x,
          int((16 << i) * width),
          b,
          expansion=expansion,
          stride1=2 if i > 0 else 1,
          normalization=normalization,
          activation=activation,
          name='conv%d' % (i + 2))
    return x

  return WideResNet(
      stack_fn=functools.partial(stack_fn, arch=arch, width=width),
      preact=True,
      model_name='{}_w{:g}_{}_{}'.format(arch, width, normalization,
                                         activation),
      input_shape=input_shape,
      pooling=pooling,
      normalization=normalization,
      activation=activation,
      num_class=num_class)


def ResNet18(**kwargs):
  return ResNetV1(arch='ResNet18', **kwargs)


def ResNet50(**kwargs):
  return ResNetV1(arch='ResNet50', **kwargs)


def ResNet101(**kwargs):
  return ResNetV1(arch='ResNet101', **kwargs)


def ResNet152(**kwargs):
  return ResNetV1(arch='ResNet152', **kwargs)


def ResNet18V2(**kwargs):
  return ResNetV2(arch='ResNet18v2', **kwargs)


def ResNet50V2(**kwargs):
  return ResNetV2(arch='ResNet50v2', **kwargs)


def ResNet101V2(**kwargs):
  return ResNetV2(arch='ResNet101v2', **kwargs)


def ResNet152V2(**kwargs):
  return ResNetV2(arch='ResNet152v2', **kwargs)


def WRN28(**kwargs):
  return WRN(arch='WRN28', **kwargs)


def WRN37(**kwargs):
  return WRN(arch='WRN37', **kwargs)
