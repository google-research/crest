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
"""Augmentation ops."""

import functools
import random

from third_party import augment_ops
from third_party import data_util as simclr_ops
from third_party import rand_augment as randaug
import tensorflow as tf


def base_augment(is_training=True, **kwargs):
  """Base (resize and crop) augmentation."""
  size, pad_size = kwargs.get('size'), int(0.125 * kwargs.get('size'))
  if is_training:
    return [
        ('resize', {
            'size': size
        }),
        ('crop', {
            'size': pad_size
        }),
    ]
  return [('resize', {'size': size})]


def crop_and_resize_augment(is_training=True, **kwargs):
  """Random crop and resize augmentation."""
  size = kwargs.get('size')
  min_scale = kwargs.get('min_scale', 0.4)
  if is_training:
    return [
        ('crop_and_resize', {
            'size': size,
            'min_scale': min_scale
        }),
    ]
  return [('resize', {'size': size})]


def jitter_augment(aug=None, is_training=True, **kwargs):
  """Color jitter augmentation."""
  if aug is None:
    aug = []
  if is_training:
    brightness = kwargs.get('brightness', 0.125)
    contrast = kwargs.get('contrast', 0.4)
    saturation = kwargs.get('saturation', 0.4)
    hue = kwargs.get('hue', 0)
    return aug + [('jitter', {
        'brightness': brightness,
        'contrast': contrast,
        'saturation': saturation,
        'hue': hue
    })]
  return aug


def cutout_augment(aug=None, is_training=True, **kwargs):
  """Cutout augmentation."""
  if aug is None:
    aug = []
  if is_training:
    scale = kwargs.get('scale', 0.5)
    return aug + [('cutout', {'scale': scale})]
  return aug


def randerase_augment(aug=None, is_training=True, **kwargs):
  """Random erase augmentation."""
  if aug is None:
    aug = []
  if is_training:
    scale = kwargs.get('scale', 0.3)
    return aug + [('randerase', {'scale': (scale, scale), 'ratio': 1.0})]
  return aug


def hflip_augment(aug=None, is_training=True, **kwargs):
  """Horizontal flip augmentation."""
  del kwargs
  if aug is None:
    aug = []
  if is_training:
    return aug + [('hflip', {})]
  return aug


def rotate90_augment(aug=None, is_training=True, **kwargs):
  """Rotation by 90 degree augmentation."""
  del kwargs
  if aug is None:
    aug = []
  if is_training:
    return aug + [('rotate90', {})]
  return aug


def rotate180_augment(aug=None, is_training=True, **kwargs):
  """Rotation by 180 degree augmentation."""
  del kwargs
  if aug is None:
    aug = []
  if is_training:
    return aug + [('rotate180', {})]
  return aug


def rotate270_augment(aug=None, is_training=True, **kwargs):
  """Rotation by 270 degree augmentation."""
  del kwargs
  if aug is None:
    aug = []
  if is_training:
    return aug + [('rotate270', {})]
  return aug


def blur_augment(aug=None, is_training=True, **kwargs):
  """Blur augmentation."""
  if aug is None:
    aug = []
  if is_training:
    prob = kwargs.get('prob', 0.5)
    return aug + [('blur', {'prob': prob})]
  return aug


def randaugment(aug=None, is_training=True, **kwargs):
  """Randaugment."""
  if aug is None:
    aug = []
  if is_training:
    num_layers = kwargs.get('num_layers', 2)
    prob_to_apply = kwargs.get('prob_to_apply', 0.5)
    magnitude = kwargs.get('magnitude', None)
    num_levels = kwargs.get('num_levels', None)
    mode = kwargs.get('mode', 'all')
    size = kwargs.get('size', None)
    return aug + [('randaug', {
        'num_layers': num_layers,
        'prob_to_apply': prob_to_apply,
        'magnitude': magnitude,
        'num_levels': num_levels,
        'size': size,
        'mode': mode
    })]
  return aug


class CutOut(object):
  """Cutout."""

  def __init__(self, scale=0.5, random_scale=False):
    self.scale = scale
    self.random_scale = random_scale

  @staticmethod
  def cutout(image, scale=0.5):
    """Applies Cutout.

    Args:
      image: A 3D tensor (width, height, depth).
      scale: A scalar for the width or height ratio for cutout region.

    Returns:
      A 3D tensor (width, height, depth) after cutout.
    """
    img_shape = tf.shape(image)
    img_height, img_width = img_shape[-3], img_shape[-2]
    img_height = tf.cast(img_height, dtype=tf.float32)
    img_width = tf.cast(img_width, dtype=tf.float32)
    cutout_size = (img_height * scale, img_width * scale)
    cutout_size = (tf.maximum(1.0,
                              cutout_size[0]), tf.maximum(1.0, cutout_size[1]))

    def _create_cutout_mask():
      height_loc = tf.round(
          tf.random.uniform(shape=[], minval=0, maxval=img_height))
      width_loc = tf.round(
          tf.random.uniform(shape=[], minval=0, maxval=img_width))

      upper_coord = (tf.maximum(0.0, height_loc - cutout_size[0] // 2),
                     tf.maximum(0.0, width_loc - cutout_size[1] // 2))
      lower_coord = (tf.minimum(img_height, height_loc + cutout_size[0] // 2),
                     tf.minimum(img_width, width_loc + cutout_size[1] // 2))
      mask_height = lower_coord[0] - upper_coord[0]
      mask_width = lower_coord[1] - upper_coord[1]

      padding_dims = ((upper_coord[0], img_height - lower_coord[0]),
                      (upper_coord[1], img_width - lower_coord[1]))
      mask = tf.zeros((mask_height, mask_width), dtype=tf.float32)
      mask = tf.pad(
          mask, tf.cast(padding_dims, dtype=tf.int32), constant_values=1.0)
      return tf.expand_dims(mask, -1)

    return _create_cutout_mask() * image

  def __call__(self, image, is_training=True):
    if is_training:
      if self.random_scale:
        scale = tf.random.uniform(shape=[], minval=0.0, maxval=self.scale)
      else:
        scale = self.scale
    return self.cutout(image, scale) if is_training else image


class RandomErase(object):
  """RandomErasing.

  Similar to Cutout, but supports various sizes and aspect ratios of rectangle.
  """

  def __init__(self, scale=(0.02, 0.3), ratio=3.3, value=0.0):
    self.scale = scale
    self.ratio = ratio
    self.value = value
    assert self.ratio >= 1

  @staticmethod
  def cutout(image, scale=(0.02, 0.3), ratio=3.3, value=0.0):
    """Applies Cutout with various sizes and aspect ratios of rectangle.

    Args:
      image: A 3D tensor (width, height, depth).
      scale: A tuple for ratio of cutout region.
      ratio: A scalar for aspect ratio.
      value: A value to fill in cutout region.

    Returns:
      A 3D tensor (width, height, depth) after cutout.
    """
    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]
    image_depth = tf.shape(image)[2]

    # Sample the center location in the image where the zero mask will be
    # applied.

    def _cutout(img):
      area = tf.cast(image_height * image_width, tf.float32)
      erase_area = tf.random.uniform(
          shape=[], minval=scale[0], maxval=scale[1]) * area
      aspect_ratio = tf.random.uniform(shape=[], minval=1, maxval=ratio)
      aspect_ratio = tf.cond(
          tf.random.uniform(shape=[]) > 0.5, lambda: aspect_ratio,
          lambda: 1.0 / aspect_ratio)
      pad_h = tf.cast(
          tf.math.round(tf.math.sqrt(erase_area * aspect_ratio)),
          dtype=tf.int32)
      pad_h = tf.minimum(pad_h, image_height - 1)
      pad_w = tf.cast(
          tf.math.round(tf.math.sqrt(erase_area / aspect_ratio)),
          dtype=tf.int32)
      pad_w = tf.minimum(pad_w, image_width - 1)

      cutout_center_height = tf.random.uniform(
          shape=[], minval=0, maxval=image_height - pad_h, dtype=tf.int32)
      cutout_center_width = tf.random.uniform(
          shape=[], minval=0, maxval=image_width - pad_w, dtype=tf.int32)

      lower_pad = cutout_center_height
      upper_pad = tf.maximum(0, image_height - cutout_center_height - pad_h)
      left_pad = cutout_center_width
      right_pad = tf.maximum(0, image_width - cutout_center_width - pad_w)

      cutout_shape = [
          image_height - (lower_pad + upper_pad),
          image_width - (left_pad + right_pad)
      ]
      padding_dims = [[lower_pad, upper_pad], [left_pad, right_pad]]
      mask = tf.pad(
          tf.zeros(cutout_shape, dtype=img.dtype),
          padding_dims,
          constant_values=1)
      mask = tf.expand_dims(mask, -1)
      mask = tf.tile(mask, [1, 1, image_depth])
      img = tf.where(
          tf.equal(mask, 0),
          tf.ones_like(img, dtype=img.dtype) * value, img)
      return img

    return _cutout(image)

  def __call__(self, image, is_training=True):
    return self.cutout(image, self.scale, self.ratio,
                       self.value) if is_training else image


class Resize(object):
  """Resize."""

  def __init__(self, size, method=tf.image.ResizeMethod.BILINEAR):
    self.size = self._check_input(size)
    self.method = method

  def _check_input(self, size):
    if isinstance(size, int):
      size = (size, size)
    elif isinstance(size, (list, tuple)) and len(size) == 1:
      size = size * 2
    else:
      raise TypeError('size must be an integer or list/tuple of integers')
    return size

  def __call__(self, image, is_training=True):
    return tf.image.resize(
        image, self.size, method=self.method) if is_training else image


class RandomCrop(object):
  """Random Crop."""

  def __init__(self, size):
    self.pad = self._check_input(size)

  def _check_input(self, size):
    """Checks pad shape.

    Args:
      size: Scalar, list or tuple for pad size.

    Returns:
      A tuple for pad size.
    """
    if isinstance(size, int):
      size = (size, size)
    elif isinstance(size, (list, tuple)):
      if len(size) == 1:
        size = tuple(size) * 2
      elif len(size) > 2:
        size = tuple(size[:2])
    else:
      raise TypeError('size must be an integer or list/tuple of integers')
    return size

  def __call__(self, image, is_training=True):
    if is_training:
      img_size = image.shape[-3:]
      image = tf.pad(
          image, [[self.pad[0]] * 2, [self.pad[1]] * 2, [0] * 2],
          mode='REFLECT')
      image = tf.image.random_crop(image, img_size)
    return image


class RandomCropAndResize(object):
  """Random crop and resize."""

  def __init__(self, size, min_scale=0.4):
    self.min_scale = min_scale
    self.size = self._check_input(size)

  def _check_input(self, size):
    """Checks input size is valid."""
    if isinstance(size, int):
      size = (size, size)
    elif isinstance(size, (list, tuple)) and len(size) == 1:
      size = size * 2
    else:
      raise TypeError('size must be an integer or list/tuple of integers')
    return size

  def __call__(self, image, is_training=True):
    if is_training:
      # crop and resize
      width = tf.random.uniform(
          shape=[],
          minval=tf.cast(image.shape[0] * self.min_scale, dtype=tf.int32),
          maxval=image.shape[0] + 1,
          dtype=tf.int32)
      size = (width, tf.minimum(width, image.shape[1]), image.shape[2])
      image = tf.image.random_crop(image, size)
      image = tf.image.resize(image, size=self.size)
    return image


class RandomFlipLeftRight(object):

  def __init__(self):
    pass

  def __call__(self, image, is_training=True):
    return tf.image.random_flip_left_right(image) if is_training else image


class ColorJitter(object):
  """Applies color jittering.

  This op is equivalent to the following:
  https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.ColorJitter
  """

  def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
    self.brightness = self._check_input(brightness)
    self.contrast = self._check_input(contrast, center=1)
    self.saturation = self._check_input(saturation, center=1)
    self.hue = self._check_input(hue, bound=0.5)

  def _check_input(self, value, center=None, bound=None):
    if bound is not None:
      value = min(value, bound)
    if center is not None:
      value = [center - value, center + value]
      if value[0] == value[1] == center:
        return None
    elif value == 0:
      return None
    return value

  def _get_transforms(self):
    """Get randomly shuffled transform ops."""
    transforms = []
    if self.brightness is not None:
      transforms.append(
          functools.partial(
              tf.image.random_brightness, max_delta=self.brightness))
    if self.contrast is not None:
      transforms.append(
          functools.partial(
              tf.image.random_contrast,
              lower=self.contrast[0],
              upper=self.contrast[1]))
    if self.saturation is not None:
      transforms.append(
          functools.partial(
              tf.image.random_saturation,
              lower=self.saturation[0],
              upper=self.saturation[1]))
    if self.hue is not None:
      transforms.append(
          functools.partial(tf.image.random_hue, max_delta=self.hue))
    random.shuffle(transforms)
    return transforms

  def __call__(self, image, is_training=True):
    if not is_training:
      return image
    for transform in self._get_transforms():
      image = transform(image)
    return image


class Rotate90(object):

  def __init__(self):
    pass

  def __call__(self, image, is_training=True):
    return tf.image.rot90(image, k=1) if is_training else image


class Rotate180(object):

  def __init__(self):
    pass

  def __call__(self, image, is_training=True):
    return tf.image.rot90(image, k=2) if is_training else image


class Rotate270(object):

  def __init__(self):
    pass

  def __call__(self, image, is_training=True):
    return tf.image.rot90(image, k=3) if is_training else image


class RandomBlur(object):

  def __init__(self, prob=0.5):
    self.prob = prob

  def __call__(self, image, is_training=True):
    if is_training:
      return image
    return simclr_ops.random_blur(
        image, image.shape[0], image.shape[1], p=self.prob)


class RandAugment(randaug.RandAugment):
  """RandAugment."""

  def __init__(self,
               num_layers=1,
               prob_to_apply=None,
               magnitude=None,
               num_levels=10,
               size=32,
               mode='all'):
    super(RandAugment, self).__init__(
        num_layers=num_layers,
        prob_to_apply=prob_to_apply,
        magnitude=magnitude,
        num_levels=num_levels)

    # override TRANSLATE_CONST
    if size == 32:
      randaug.TRANSLATE_CONST = 10.
    elif size == 96:
      randaug.TRANSLATE_CONST = 30.
    elif size == 128:
      randaug.TRANSLATE_CONST = 40.
    elif size == 256:
      randaug.TRANSLATE_CONST = 100.
    else:
      randaug.TRANSLATE_CONST = int(0.3 * size)
    assert mode.upper() in [
        'ALL', 'COLOR', 'GEO', 'CUTOUT'
    ], 'RandAugment mode should be `All`, `COLOR` or `GEO`'
    self.mode = mode.upper()
    self._register_ops()
    if mode.upper() == 'CUTOUT':
      self.cutout_ops = CutOut(scale=0.5, random_scale=True)

  def _generate_branch_fn(self, image, level):
    branch_fns = []
    for augment_op_name in self.ra_ops:
      augment_fn = augment_ops.NAME_TO_FUNC[augment_op_name]
      level_to_args_fn = randaug.LEVEL_TO_ARG[augment_op_name]

      def _branch_fn(image=image,
                     augment_fn=augment_fn,
                     level_to_args_fn=level_to_args_fn):
        args = [image] + list(level_to_args_fn(level))
        return augment_fn(*args)

      branch_fns.append(_branch_fn)
    return branch_fns

  def _apply_one_layer(self, image):
    """Applies one level of augmentation to the image."""
    level = self._get_level()
    branch_index = tf.random.uniform(
        shape=[], maxval=len(self.ra_ops), dtype=tf.int32)
    num_concat = image.shape[2] // 3
    images = tf.split(image, num_concat, axis=-1)
    aug_images = []
    for image_slice in images:
      branch_fns = self._generate_branch_fn(image_slice, level)
      # pylint: disable=cell-var-from-loop
      aug_image_slice = tf.switch_case(
          branch_index, branch_fns, default=lambda: image_slice)
      aug_images.append(aug_image_slice)
    aug_image = tf.concat(aug_images, axis=-1)
    if self.prob_to_apply is not None:
      return tf.cond(
          tf.random.uniform(shape=[], dtype=tf.float32) < self.prob_to_apply,
          lambda: aug_image, lambda: image)
    else:
      return aug_image

  def _register_ops(self):
    if self.mode == 'ALL':
      self.ra_ops = [
          'AutoContrast',
          'Equalize',
          'Posterize',
          'Solarize',
          'Color',
          'Contrast',
          'Brightness',
          'Identity',
          'Invert',
          'Sharpness',
          'SolarizeAdd',
      ]
      self.ra_ops += [
          'Rotate',
          'ShearX',
          'ShearY',
          'TranslateX',
          'TranslateY',
      ]
    elif self.mode == 'CUTOUT':
      self.ra_ops = [
          'AutoContrast',
          'Equalize',
          'Posterize',
          'Solarize',
          'Color',
          'Contrast',
          'Brightness',
          'Identity',
          'Invert',
          'Sharpness',
          'SolarizeAdd',
      ]
      self.ra_ops += [
          'Rotate',
          'ShearX',
          'ShearY',
          'TranslateX',
          'TranslateY',
      ]
    elif self.mode == 'COLOR':
      self.ra_ops = [
          'AutoContrast',
          'Equalize',
          'Posterize',
          'Solarize',
          'Color',
          'Contrast',
          'Brightness',
          'Identity',
          'Invert',
          'Sharpness',
          'SolarizeAdd',
      ]
    elif self.mode == 'GEO':
      self.ra_ops = [
          'Rotate',
          'ShearX',
          'ShearY',
          'TranslateX',
          'TranslateY',
          'Identity',
      ]
    else:
      raise NotImplementedError

  def wrap(self, image):
    image += tf.constant(1.0, image.dtype)
    image *= tf.constant(255.0 / 2.0, image.dtype)
    image = tf.saturate_cast(image, tf.uint8)
    return image

  def unwrap(self, image):
    image = tf.cast(image, tf.float32)
    image /= tf.constant(255.0 / 2.0, image.dtype)
    image -= tf.constant(1.0, image.dtype)
    return image

  def _apply_cutout(self, image):
    # Cutout assumes pixels are in [-1, 1].
    aug_image = self.unwrap(image)
    aug_image = self.cutout_ops(aug_image)
    aug_image = self.wrap(aug_image)
    if self.prob_to_apply is not None:
      return tf.cond(
          tf.random.uniform(shape=[], dtype=tf.float32) < self.prob_to_apply,
          lambda: aug_image, lambda: image)
    else:
      return aug_image

  def __call__(self, image, is_training=True):
    if not is_training:
      return image
    image = self.wrap(image)
    if self.mode == 'CUTOUT':
      for _ in range(self.num_layers):
        # Makes an exception for cutout.
        image = tf.cond(
            tf.random.uniform(shape=[], dtype=tf.float32) < tf.divide(
                tf.constant(1.0), tf.cast(
                    len(self.ra_ops) + 1, dtype=tf.float32)),
            lambda: self._apply_cutout(image),
            lambda: self._apply_one_layer(image))
      return self.unwrap(image)
    else:
      for _ in range(self.num_layers):
        image = self._apply_one_layer(image)
      return self.unwrap(image)
