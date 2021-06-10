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
"""Augmentation functions."""

import functools

from data import augment_ops
from immutabledict import immutabledict

_SUPPORTED_AUG_SHORTCUT = immutabledict({
    'd': '',
    'x': 'x',
    'dh': 'hflip',
    'dhc': 'hflip+cutout0.5',
    'dhrac': 'hflip+randaug2-0.5---all+cutout0.5',
})

_SUPPORTED_AUG_OPS = immutabledict({
    'base': functools.partial(augment_ops.base_augment),
    'hflip': functools.partial(augment_ops.hflip_augment),
    'jitter': functools.partial(augment_ops.jitter_augment),
    'blur': functools.partial(augment_ops.blur_augment),
    'cutout': functools.partial(augment_ops.cutout_augment),
    'cnr': functools.partial(augment_ops.crop_and_resize_augment),
    'crop_and_resize': functools.partial(augment_ops.crop_and_resize_augment),
    'randerase': functools.partial(augment_ops.randerase_augment),
    'rotate90': functools.partial(augment_ops.rotate90_augment),
    'rotate180': functools.partial(augment_ops.rotate180_augment),
    'rotate270': functools.partial(augment_ops.rotate270_augment),
    'randaug': functools.partial(augment_ops.randaugment),
})


def apply_augment(image, ops_list=None):
  """Applies Augmentation Sequence.

  Args:
    image: 3D tensor of (height, width, channel).
    ops_list: List of augmentation operation returned by compose_augment_seq.

  Returns:
    List of augmented images.
  """
  if ops_list is None:
    return (image,)
  if not isinstance(ops_list, (tuple, list)):
    ops_list = [ops_list]

  def _apply_augment(image, ops):
    for op in ops:
      image = op(image)
    return image

  return tuple([_apply_augment(image, ops) for ops in ops_list])


def compose_augment_seq(aug_list, is_training=False):
  """Composes Augmentation Sequence.

  Args:
    aug_list: List of tuples (aug_type, kwargs).
    is_training: Boolean, if True applies stochastic augmentation.

  Returns:
    List of augmentation ops.
  """
  return [
      generate_augment_ops(aug_type, is_training=is_training, **kwargs)
      for aug_type, kwargs in aug_list
  ]


def generate_augment_ops(aug_type, is_training=False, **kwargs):
  """Generates Augmentation Operators.

  Args:
    aug_type: Augmentation type.
    is_training: Boolean, if True applies stochastic augmentation.
    **kwargs: Additional arguments.

  Returns:
    An augmentation function.
  """
  registered_ops = [
      'resize', 'crop', 'crop_and_resize', 'hflip', 'blur', 'jitter', 'cutout',
      'randerase', 'randaug', 'rotate90', 'rotate180', 'rotate270'
  ]
  assert aug_type.lower() in registered_ops

  if aug_type.lower() == 'randerase':
    scale = kwargs.get('scale', (0.02, 0.3))
    ratio = kwargs.get('ratio', 3.3)
    value = kwargs.get('value', 0.5)
    tx_op = augment_ops.RandomErase(scale=scale, ratio=ratio, value=value)

  elif aug_type.lower() == 'randaug':
    num_layers = kwargs.get('num_layers', 2)
    prob_to_apply = kwargs.get('prob_to_apply', 0.5)
    magnitude = kwargs.get('magnitude', None)
    num_levels = kwargs.get('num_levels', None)
    mode = kwargs.get('mode', 'all')
    size = kwargs.get('size', None)
    tx_op = augment_ops.RandAugment(
        num_layers=num_layers,
        prob_to_apply=prob_to_apply,
        magnitude=magnitude,
        num_levels=num_levels,
        size=size,
        mode=mode)

  elif aug_type.lower() == 'cutout':
    scale = kwargs.get('scale', 0.5)
    tx_op = augment_ops.CutOut(scale=scale)

  elif aug_type.lower() == 'jitter':
    brightness = kwargs.get('brightness', 0.125)
    contrast = kwargs.get('contrast', 0.4)
    saturation = kwargs.get('saturation', 0.4)
    hue = kwargs.get('hue', 0)
    tx_op = augment_ops.ColorJitter(
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue)

  elif aug_type.lower() == 'crop':
    size = kwargs.get('size', 0)
    tx_op = augment_ops.RandomCrop(size=size)

  elif aug_type.lower() == 'hflip':
    tx_op = augment_ops.RandomFlipLeftRight()

  elif aug_type.lower() == 'resize':
    size = kwargs.get('size', 256)
    tx_op = augment_ops.Resize(size)

  elif aug_type.lower() == 'crop_and_resize':
    size = kwargs.get('size', 224)
    min_scale = kwargs.get('min_scale', 0.4)
    tx_op = augment_ops.RandomCropAndResize(size=size, min_scale=min_scale)

  elif aug_type.lower() == 'rotate90':
    tx_op = augment_ops.Rotate90()

  elif aug_type.lower() == 'rotate180':
    tx_op = augment_ops.Rotate180()

  elif aug_type.lower() == 'rotate270':
    tx_op = augment_ops.Rotate270()

  elif aug_type.lower() == 'blur':
    prob = kwargs.get('prob', 0.5)
    tx_op = augment_ops.RandomBlur(prob=prob)

  return functools.partial(tx_op, is_training=is_training)


def retrieve_augment(aug_list, **kwargs):
  """Retrieves Augmentation Sequences.

  Args:
    aug_list: Nested list of tuples of (aug_type, kwargs)
    **kwargs: Additional arguments.

  Returns:
    List of augmentation lists.
  """

  def _get_augment_from_shortcut(aug_name):
    if aug_name not in _SUPPORTED_AUG_SHORTCUT:
      return aug_name
    return _SUPPORTED_AUG_SHORTCUT[aug_name]

  def _get_augment_args(aug_name, **kwargs):
    aug_args = kwargs
    if aug_name.startswith('cutout'):
      scale = aug_name.replace('cutout', '')
      if scale:
        aug_args['scale'] = float(scale)
      aug_name = 'cutout'
    elif aug_name.startswith('randaug'):
      # [num_layers]-[prob_to_apply]-[magnitude]-[num_layers]-[mode]
      args = aug_name.replace('randaug', '').split('-')
      if len(args) != 5:
        raise ValueError('Randaug requires 5 arguments of # layers, probability'
                         'to apply, magnitude, # levels, and mode.')
      aug_args['num_layers'] = int(args[0]) if args[0] else 2
      aug_args['prob_to_apply'] = float(args[1]) if args[1] else 0.5
      aug_args['magnitude'] = float(args[2]) if args[2] else None
      aug_args['num_levels'] = int(args[3]) if args[3] else None
      aug_args['mode'] = args[4] if args[4] else 'all'
      aug_name = 'randaug'
    elif aug_name.startswith('blur'):
      prob = aug_name.replace('blur', '')
      if prob:
        aug_args['prob'] = float(prob)
      aug_name = 'blur'
    elif aug_name.startswith('jitter'):
      augs = aug_name.replace('jitter', '')
      if augs:
        augs_list = filter(None, augs.split('_'))
        for aug in augs_list:
          if aug[0] == 'b':
            aug_args['brightness'] = float(aug[1:])
          elif aug[0] == 'c':
            aug_args['contrast'] = float(aug[1:])
          elif aug[0] == 's':
            aug_args['saturation'] = float(aug[1:])
          elif aug[0] == 'h':
            aug_args['hue'] = float(aug[1:])
      aug_name = 'jitter'
    elif aug_name.startswith('cnr'):
      min_scale = aug_name.replace('cnr', '')
      if min_scale:
        aug_args['min_scale'] = float(min_scale)
      aug_name = 'cnr'
    elif aug_name.startswith('crop_and_resize'):
      min_scale = aug_name.replace('crop_and_resize', '')
      if min_scale:
        aug_args['min_scale'] = float(min_scale)
      aug_name = 'crop_and_resize'

    return aug_name, aug_args

  def _retrieve_augment(aug_name, is_training=True):
    if aug_name in _SUPPORTED_AUG_OPS:
      return functools.partial(
          _SUPPORTED_AUG_OPS[aug_name], is_training=is_training)
    else:
      raise NotImplementedError

  # Retrieve augmentation ops
  aug_fn_list = []
  for aug_names in aug_list:
    aug_names = _get_augment_from_shortcut(aug_names)
    # chaining from the base augmentation
    if aug_names == 'x':
      # no augmentation
      aug_fn = _retrieve_augment('base', is_training=False)(**kwargs)
    else:
      aug_fn = _retrieve_augment('base', is_training=True)(**kwargs)
      aug_name_list = filter(None, aug_names.split('+'))
      for aug_name in aug_name_list:
        aug_name, aug_args = _get_augment_args(aug_name, **kwargs)
        aug_fn = _retrieve_augment(
            aug_name, is_training=True)(
                aug=aug_fn, **aug_args)
    aug_fn_list.append(aug_fn)
    if len(aug_fn_list) == 1:
      # Generates test augmentation.
      if aug_names == 'x':
        # No augmentation.
        test_aug_fn = _retrieve_augment('base', is_training=False)(**kwargs)
      else:
        test_aug_fn = _retrieve_augment('base', is_training=False)(**kwargs)
        aug_name_list = filter(None, aug_names.split('+'))
        for aug_name in aug_name_list:
          aug_name, aug_args = _get_augment_args(aug_name, **kwargs)
          test_aug_fn = _retrieve_augment(
              aug_name, is_training=False)(
                  aug=test_aug_fn, **aug_args)
  return aug_fn_list + [test_aug_fn]
