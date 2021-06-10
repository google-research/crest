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
"""Class to load CIFAR dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from data import augment as augment_lib
from data import data_util
import numpy as np
import tensorflow as tf

CIFAR_DIR = os.path.join(os.getenv('ML_DATA'), 'cifar')
CIFAR_LT_DIR = os.path.join(os.getenv('ML_DATA'), 'cifar-lt')
CIFAR_DARP_DIR = os.path.join(os.getenv('ML_DATA'), 'cifar-darp')


class CIFAR10(object):
  """CIFAR10 dataloader."""

  def __init__(self):
    self.load_raw_data()

  def load_raw_data(self):
    """Loads CIFAR10 raw data."""
    self.data_name = 'cifar10'
    (x_train, y_train) = data_util.load_tfrecord(
        os.path.join(CIFAR_DIR, 'cifar10-train.tfrecord'))
    (x_test, y_test) = data_util.load_tfrecord(
        os.path.join(CIFAR_DIR, 'cifar10-test.tfrecord'))
    self.x_train = x_train
    self.y_train = y_train
    self.x_test = x_test
    self.y_test = y_test
    self.x_unlab = x_train
    self.y_unlab = y_train
    self.num_class = 10

  def get_prefix(self,
                 fold=1,
                 num_labeled_per_class=10,
                 augment=None,
                 is_balanced=False):
    """Gets prefix for file name."""
    if num_labeled_per_class > 0:
      self.fname = '{}.{}.{}@{}'.format(
          self.data_name, '.'.join(['_'.join(aug) for aug in augment]), fold,
          num_labeled_per_class * self.num_class)
      if not is_balanced:
        self.fname = '{}.{}'.format(self.fname, 'imbalance')
    else:
      self.fname = '{}.{}@full'.format(self.data_name, augment[0][0])

  def get_split(self, fold=1, num_labeled_per_class=10, is_balanced=True):
    """Gets labeled and unlabeled data split."""
    np.random.seed(fold)
    if is_balanced:
      class_id = {}
      for i, y in enumerate(self.y_train):
        if y not in class_id:
          class_id[y] = []
        class_id[y].append(i)
      labeled_idx = []
      for c in sorted(class_id):
        np.random.shuffle(class_id[c])
        labeled_idx += class_id[c][:num_labeled_per_class]
      self.x_train = self.x_train[labeled_idx]
      self.y_train = self.y_train[labeled_idx]
    else:
      perm_idx = np.random.permutation(len(self.y_train))
      self.x_train = self.x_train[perm_idx][:num_labeled_per_class *
                                            self.n_class]
      self.y_train = self.y_train[perm_idx][:num_labeled_per_class *
                                            self.n_class]

  def load_dataset(self,
                   fold=1,
                   num_labeled_per_class=10,
                   is_balanced=True,
                   input_shape=(32, 32, 3),
                   augment=None,
                   batch_size=64,
                   batch_size_unlab=0,
                   num_workers=4,
                   strategy=None,
                   **kwargs):
    """Loads dataset."""
    del kwargs

    # Generate labeled data.
    if num_labeled_per_class > 0:
      self.get_split(
          fold=fold,
          num_labeled_per_class=num_labeled_per_class,
          is_balanced=is_balanced)

    # Construct dataset
    train_data = (self.x_train, self.y_train,
                  np.expand_dims(np.arange(len(self.y_train)), axis=1))
    test_data = (self.x_test, self.y_test,
                 np.expand_dims(np.arange(len(self.y_test)), axis=1))
    if len(train_data[0]) < batch_size:
      # if number of examples is less than batch size,
      # we increase the number by replicating
      multiple = int(2 * (np.math.ceil(batch_size / len(train_data[0]))))
      train_data = (np.concatenate([train_data[0] for _ in range(multiple)],
                                   axis=0),
                    np.concatenate([train_data[1] for _ in range(multiple)],
                                   axis=0),
                    np.concatenate([train_data[2] for _ in range(multiple)],
                                   axis=0))
    train_set = data_util.ImageFromMemory(
        data=train_data, input_shape=input_shape)
    test_set = data_util.ImageFromMemory(
        data=test_data, input_shape=input_shape)

    aug_args = {'size': input_shape[0]}
    augs, augs_for_prefix = [], []
    for aug in augment:
      aug, num_aug = aug
      if num_aug == 0:
        continue
      if len(aug) == num_aug:
        augs_for_prefix.append(aug)
      elif len(aug) < num_aug:
        assert len(aug) == 1, (
            'cannot have multiple aug types if num_aug is larger than the '
            'number of aug types')
        augs_for_prefix.append(['{}{}'.format(num_aug, a) for a in aug])
        aug *= num_aug
      else:
        augs_for_prefix.append(aug)
        num_aug = len(aug)
      augs.append(augment_lib.retrieve_augment(aug, **aug_args))

    self.get_prefix(
        fold=fold,
        num_labeled_per_class=num_labeled_per_class,
        augment=augs_for_prefix,
        is_balanced=is_balanced)

    train_loader = train_set.input_fn(
        is_training=True,
        batch_size=batch_size,
        aug_list=augs[0][:-1],
        dtype=tf.float32,
        num_cores=num_workers,
        strategy=strategy)
    test_loader = test_set.input_fn(
        is_training=False,
        batch_size=100,
        aug_list=augs[0][-1],
        dtype=tf.float32,
        num_cores=max(num_workers // 4, 1),
        strategy=strategy)

    # semi-supervised setting
    if batch_size_unlab > 0:
      unlab_data = (self.x_unlab, self.y_unlab,
                    np.expand_dims(np.arange(len(self.y_unlab)), axis=1))
      unlab_set = data_util.ImageFromMemory(
          data=unlab_data, input_shape=input_shape)
      augs_unlab = []
      for sublist in augs[1:]:
        for item in sublist[:-1]:
          augs_unlab.append(item)
      unlab_loader = unlab_set.input_fn(
          is_training=True,
          batch_size=batch_size_unlab,
          aug_list=augs_unlab,
          dtype=tf.float32,
          num_cores=num_workers,
          strategy=strategy)
      return [
          tf.data.Dataset.zip((train_loader, unlab_loader)), None, test_loader
      ]
    return [train_loader, None, test_loader]


class CIFAR100(CIFAR10):
  """CIFAR100 dataset."""

  def load_raw_data(self):
    """Loads CIFAR100 raw data."""
    self.data_name = 'cifar100'
    (x_train, y_train) = data_util.load_tfrecord(
        os.path.join(CIFAR_DIR, 'cifar100-train.tfrecord'))
    (x_test, y_test) = data_util.load_tfrecord(
        os.path.join(CIFAR_DIR, 'cifar100-test.tfrecord'))
    self.x_train = x_train
    self.y_train = y_train
    self.x_test = x_test
    self.y_test = y_test
    self.x_unlab = x_train
    self.y_unlab = y_train
    self.num_class = 100


class CIFAR10LT(object):
  """CIFAR10 long-tail data loader."""

  def __init__(self, class_im_ratio):
    self.load_raw_data(class_im_ratio)

  def load_raw_data(self, class_im_ratio):
    """Loads CIFAR10 long-tail raw data."""
    self.data_name = 'cifar10lt@{}'.format(class_im_ratio)
    dir_path = os.path.join(CIFAR_LT_DIR,
                            'cifar-10-data-im-{}'.format(class_im_ratio))
    data_shape = self.load_raw_data_shape()
    (x_train, y_train) = data_util.load_tfrecord(
        os.path.join(dir_path, 'train.tfrecords'), is_raw=True, **data_shape)
    (x_test, y_test) = data_util.load_tfrecord(
        os.path.join(dir_path, 'eval.tfrecords'), is_raw=True, **data_shape)
    self.x_train = x_train
    self.y_train = y_train
    self.x_test = x_test
    self.y_test = y_test
    self.x_unlab = x_train
    self.y_unlab = y_train
    self.x_unlab_test = x_train
    self.y_unlab_test = y_train
    self.class_im_ratio = class_im_ratio
    self.num_class = 10

    # The class distribution is long-tailed. For K classes, the most major class
    # has N_1 samples, and the most minor class has N_K = N_1 * class_im_ratio
    # samples. The k-th class has N_k = N_1 * class_im_ratio^((k - 1) / (K - 1))
    # samples. Reference: DARP (https://arxiv.org/pdf/2007.08844.pdf)
    e = tf.cast(tf.range(self.num_class), tf.float32) / (self.num_class - 1)
    base = tf.ones_like(e) * self.class_im_ratio
    self.gt_p_data = tf.pow(base, e)
    self.gt_p_data /= tf.math.reduce_sum(self.gt_p_data)
    print('ground truth classs discribution: {}'.format(self.gt_p_data))

  def load_raw_data_shape(self):
    return {'depth': 3, 'height': 32, 'width': 32}

  def get_prefix(self,
                 fold=1,
                 augment=None,
                 percent_labeled_per_class=0.1,
                 update_mode='distribution',
                 alpha=3):
    """Gets prefix for file name."""
    self.fname = '{}.{}.{}@{}'.format(
        self.data_name, '.'.join(['_'.join(aug) for aug in augment]), fold,
        percent_labeled_per_class)
    if update_mode == 'distribution':
      self.fname = '{}_{}_{}'.format(self.fname, update_mode, int(alpha))
    elif update_mode == 'all':
      self.fname = '{}_{}'.format(self.fname, update_mode)
    else:
      raise NotImplementedError

  def get_split(self,
                fold=1,
                percent_labeled_per_class=0.1,
                update_mode='distribution',
                alpha=3,
                pseudo_label_list=None):
    """Gets labeled and unlabeled data split."""
    np.random.seed(fold)

    class_id = {}
    for i, y in enumerate(self.y_train):
      if y not in class_id:
        class_id[y] = []
      class_id[y].append(i)
    labeled_idx = []
    unlabeled_idx = []
    for c in sorted(class_id):
      np.random.shuffle(class_id[c])
      num_labeled_this_class = int(
          np.ceil(len(class_id[c]) * percent_labeled_per_class))
      print('class {} has {} images, {} labels'.format(c, len(class_id[c]),
                                                       num_labeled_this_class))
      labeled_idx += class_id[c][:num_labeled_this_class]
      unlabeled_idx += class_id[c][num_labeled_this_class:]

    if pseudo_label_list:
      x_picked = []
      y_picked = []
      if update_mode == 'distribution':
        sample_rate = self.gt_p_data[::-1] / self.gt_p_data[0]
        for c in range(self.num_class):
          num_picked = int(
              len(pseudo_label_list[c]) *
              np.math.pow(sample_rate[c], 1 / alpha))
          idx_picked = pseudo_label_list[c][:num_picked]
          idx_picked = [unlabeled_idx[idx] for idx in idx_picked]
          x_picked.append(self.x_train[idx_picked])
          y_picked.append(np.ones_like(self.y_train[idx_picked]) * c)
          print('class {} is added {} pseudo images'.format(c, len(idx_picked)))
      elif update_mode == 'all':
        for c in range(self.num_class):
          num_picked = len(pseudo_label_list[c])
          idx_picked = pseudo_label_list[c][:num_picked]
          idx_picked = [unlabeled_idx[idx] for idx in idx_picked]
          x_picked.append(self.x_train[idx_picked])
          y_picked.append(np.ones_like(self.y_train[idx_picked]) * c)
          print('class {} is added {} pseudo images'.format(c, len(idx_picked)))
      else:
        raise NotImplementedError
      x_picked.append(self.x_train[labeled_idx])
      y_picked.append(self.y_train[labeled_idx])
      self.x_train = np.concatenate(x_picked, axis=0)
      self.y_train = np.concatenate(y_picked, axis=0)
      print('update training set with mode {}'.format(update_mode))
    else:
      self.x_train = self.x_train[labeled_idx]
      self.y_train = self.y_train[labeled_idx]
      print('not update')
    print('{} train set images in total'.format(len(self.x_train)))

    self.x_unlab_test = self.x_unlab_test[unlabeled_idx]
    self.y_unlab_test = self.y_unlab_test[unlabeled_idx]

  def load_dataset(self,
                   fold=1,
                   num_labeled_per_class=10,
                   input_shape=(32, 32, 3),
                   augment=None,
                   batch_size=64,
                   batch_size_unlab=0,
                   num_workers=4,
                   strategy=None,
                   **kwargs):
    """Loads dataset."""
    percent_labeled_per_class = kwargs.get('percent_labeled_per_class', 0.1)
    update_mode = kwargs.get('update_mode', 'all')
    alpha = kwargs.get('alpha', 3)
    pseudo_label_list = kwargs.get('pseudo_label_list', None)

    # Generate labeled data.
    if num_labeled_per_class > 0:
      self.get_split(
          fold=fold,
          percent_labeled_per_class=percent_labeled_per_class,
          update_mode=update_mode,
          alpha=alpha,
          pseudo_label_list=pseudo_label_list)

    # Construct dataset.
    train_data = (self.x_train, self.y_train,
                  np.expand_dims(np.arange(len(self.y_train)), axis=1))
    test_data = (self.x_test, self.y_test,
                 np.expand_dims(np.arange(len(self.y_test)), axis=1))
    unlab_test_data = (self.x_unlab_test, self.y_unlab_test,
                       np.expand_dims(
                           np.arange(len(self.y_unlab_test)), axis=1))
    if len(train_data[0]) < batch_size:
      # if number of examples is less than batch size,
      # we increase the number by replicating.
      multiple = int(2 * (np.math.ceil(batch_size / len(train_data[0]))))
      train_data = (np.concatenate([train_data[0] for _ in range(multiple)],
                                   axis=0),
                    np.concatenate([train_data[1] for _ in range(multiple)],
                                   axis=0),
                    np.concatenate([train_data[2] for _ in range(multiple)],
                                   axis=0))
    train_set = data_util.ImageFromMemory(
        data=train_data, input_shape=input_shape)
    test_set = data_util.ImageFromMemory(
        data=test_data, input_shape=input_shape)
    unlab_test_set = data_util.ImageFromMemory(
        data=unlab_test_data, input_shape=input_shape)

    aug_args = {'size': input_shape[0]}
    augs, augs_for_prefix = [], []
    for aug in augment:
      aug, num_aug = aug
      if num_aug == 0:
        continue
      if len(aug) == num_aug:
        augs_for_prefix.append(aug)
      elif len(aug) < num_aug:
        assert len(aug) == 1, (
            'cannot have multiple aug types if num_aug is larger than the '
            'number of aug types')
        augs_for_prefix.append(['{}{}'.format(num_aug, a) for a in aug])
        aug *= num_aug
      else:
        augs_for_prefix.append(aug)
        num_aug = len(aug)
      augs.append(augment_lib.retrieve_augment(aug, **aug_args))

    self.get_prefix(
        fold=fold,
        augment=augs_for_prefix,
        percent_labeled_per_class=percent_labeled_per_class,
        update_mode=update_mode,
        alpha=alpha)

    train_loader = train_set.input_fn(
        is_training=True,
        batch_size=batch_size,
        aug_list=augs[0][:-1],
        dtype=tf.float32,
        num_cores=num_workers,
        strategy=strategy)
    test_loader = test_set.input_fn(
        is_training=False,
        batch_size=100,
        aug_list=augs[0][-1],
        dtype=tf.float32,
        num_cores=max(num_workers // 4, 1),
        strategy=strategy)
    unlab_test_loader = unlab_test_set.input_fn(
        is_training=False,
        batch_size=100,
        aug_list=augs[0][-1],
        dtype=tf.float32,
        num_cores=max(num_workers // 4, 1),
        strategy=strategy)

    # Semi-supervised setting.
    if batch_size_unlab > 0:
      unlab_data = (self.x_unlab, self.y_unlab,
                    np.expand_dims(np.arange(len(self.y_unlab)), axis=1))
      unlab_set = data_util.ImageFromMemory(
          data=unlab_data, input_shape=input_shape)
      augs_unlab = []
      for sublist in augs[1:]:
        for item in sublist[:-1]:
          augs_unlab.append(item)
      unlab_loader = unlab_set.input_fn(
          is_training=True,
          batch_size=batch_size_unlab,
          aug_list=augs_unlab,
          dtype=tf.float32,
          num_cores=num_workers,
          strategy=strategy)
      return [
          tf.data.Dataset.zip((train_loader, unlab_loader)), unlab_test_loader,
          test_loader
      ]
    return [train_loader, unlab_test_loader, test_loader]


class CIFAR100LT(CIFAR10LT):
  """CIFAR100 long-tail data loader."""

  def load_raw_data(self, class_im_ratio):
    """Loads CIFAR100 long-tail raw data."""
    self.data_name = 'cifar100lt@{}'.format(class_im_ratio)
    dir_path = os.path.join(CIFAR_LT_DIR,
                            'cifar-100-data-im-{}'.format(class_im_ratio))
    data_shape = {'depth': 3, 'height': 32, 'width': 32}
    (x_train, y_train) = data_util.load_tfrecord(
        os.path.join(dir_path, 'train.tfrecords'), is_raw=True, **data_shape)
    (x_test, y_test) = data_util.load_tfrecord(
        os.path.join(dir_path, 'eval.tfrecords'), is_raw=True, **data_shape)
    self.x_train = x_train
    self.y_train = y_train
    self.x_test = x_test
    self.y_test = y_test
    self.x_unlab = x_train
    self.y_unlab = y_train
    self.x_unlab_test = x_train
    self.y_unlab_test = y_train
    self.num_class = 100
    self.class_im_ratio = class_im_ratio

    e = tf.cast(tf.range(self.num_class), tf.float32) / (self.num_class - 1)
    base = tf.ones_like(e) * self.class_im_ratio
    self.gt_p_data = tf.pow(base, e)
    self.gt_p_data /= tf.math.reduce_sum(self.gt_p_data)
    print('ground truth classs discribution: {}'.format(self.gt_p_data))


class CIFAR10LTDARP(CIFAR10LT):
  """CIFAR10 long-tail data loader following DARP setting.

  Jaehyung Kim, Youngbum Hur, Sejun Park, Eunho Yang,
  Sung Ju Hwang, and Jinwoo Shin.
  Distribution Aligning Refinery of Pseudo-label for
  Imbalanced Semi-supervised Learning. (https://arxiv.org/abs/2007.08844)
  """

  def load_raw_data(self, class_im_ratio):
    self.data_name = 'cifar10ltdarp{}'.format(class_im_ratio)
    data_shape = {'depth': 3, 'height': 32, 'width': 32}
    (x_train, y_train) = data_util.load_tfrecord(
        os.path.join(
            CIFAR_DARP_DIR,
            'cifar10-{}-train-labeled.tfrecords'.format(int(class_im_ratio))),
        is_raw=True,
        **data_shape)
    (x_unlab_test, y_unlab_test) = data_util.load_tfrecord(
        os.path.join(
            CIFAR_DARP_DIR,
            'cifar10-{}-train-unlabeled.tfrecords'.format(int(class_im_ratio))),
        is_raw=True,
        **data_shape)
    (x_test, y_test) = data_util.load_tfrecord(
        os.path.join(CIFAR_DARP_DIR, 'cifar10-test.tfrecords'),
        is_raw=True,
        **data_shape)
    self.x_train = x_train
    self.y_train = y_train
    self.x_test = x_test
    self.y_test = y_test
    self.x_unlab_test = x_unlab_test
    self.y_unlab_test = y_unlab_test
    self.x_unlab = np.concatenate([x_train, x_unlab_test], axis=0)
    self.y_unlab = np.concatenate([y_train, y_unlab_test], axis=0)
    self.num_class = 10
    self.class_im_ratio = 1 / class_im_ratio

    pow_value = tf.cast(tf.range(self.num_class), tf.float32) / (
        self.num_class - 1)
    base = tf.ones_like(pow_value) * self.class_im_ratio
    self.gt_p_data = tf.pow(base, pow_value)
    self.gt_p_data /= tf.math.reduce_sum(self.gt_p_data)
    print('ground truth classs discribution: {}'.format(self.gt_p_data))

  def get_split(self,
                fold=1,
                percent_labeled_per_class=0.1,
                update_mode='distribution',
                alpha=3,
                pseudo_label_list=None):
    if pseudo_label_list is not None:
      x_picked = []
      y_picked = []
      if update_mode == 'distribution':
        mu = np.math.pow(self.class_im_ratio, 1 / 9)
        for c in range(self.num_class):
          num_picked = int(
              len(pseudo_label_list[c]) *
              np.math.pow(np.math.pow(mu, 9 - c), 1 / alpha))
          idx_picked = pseudo_label_list[c][:num_picked]
          x_picked.append(self.x_unlab_test[idx_picked])
          y_picked.append(np.ones_like(self.y_unlab_test[idx_picked]) * c)
          print('class {} is added {} pseudo labels'.format(c, num_picked))
        x_picked.append(self.x_train)
        y_picked.append(self.y_train)
        self.x_train = np.concatenate(x_picked, axis=0)
        self.y_train = np.concatenate(y_picked, axis=0)
      else:
        raise NotImplementedError
    else:
      print('not update')
    print('{} train set images in total'.format(len(self.x_train)))
