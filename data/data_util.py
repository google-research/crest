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
"""Utilities to load data."""

from absl import logging
from data import augment
import numpy as np
import tensorflow as tf

# means and stds are from ImageNet dataset.
CHANNEL_MEANS = [0.485, 0.456, 0.406]
CHANNEL_STDS = [0.229, 0.224, 0.225]


def parse_tfrecord_file(filenames, is_raw=False, **kwargs):
  """Parses tfrecord file."""
  dataset = tf.data.TFRecordDataset(filenames)

  # depth, height, width
  depth = kwargs.get('depth', None)
  height = kwargs.get('height', None)
  width = kwargs.get('width', None)

  def _record_parse(serialized_example):
    features = tf.io.parse_single_example(
        serialized_example,
        features={
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64)
        })
    image = tf.image.decode_image(features['image'])
    return image, features['label']

  def _parse_raw_record(serialized_example):
    features = tf.io.parse_single_example(
        serialized_example,
        features={
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64)
        })
    image = tf.io.decode_raw(features['image'], tf.uint8)
    image.set_shape([depth * height * width])
    image = tf.transpose(tf.reshape(image, [depth, height, width]), [1, 2, 0])
    return image, features['label']

  return dataset.map(_parse_raw_record) if is_raw else dataset.map(
      _record_parse)


def load_tfrecord(filenames, **kwargs):
  """Loads tfrecord."""
  dataset = parse_tfrecord_file(filenames, **kwargs)
  data = []
  labels = []
  for example, label in dataset:
    data.append(example.numpy())
    labels.append(label.numpy())
  return (np.stack(data, axis=0), np.stack(labels, axis=0))


class BasicImageProcess():
  """Basic image process class."""

  def __init__(self, input_shape=(256, 256, 3)):
    self.input_shape = input_shape

  def image_normalize(self, image, do_mean=True, do_std=True):
    if do_mean:
      means = tf.broadcast_to(CHANNEL_MEANS, tf.shape(image))
      image = image - means
    if do_std:
      stds = tf.broadcast_to(CHANNEL_STDS, tf.shape(image))
      image = image / stds
    return image

  def preprocess_image(self,
                       image,
                       dtype=tf.float32,
                       aug_ops_list=None,
                       **kwargs):
    """Preprocesses an image."""
    del kwargs

    image = (2.0 / 255.0) * tf.cast(image, dtype) - 1.0
    image = tf.reshape(image, shape=tf.stack(self.input_shape))
    images = augment.apply_augment(image, ops_list=aug_ops_list)
    return images

  def parse_record_fn(self, raw_record, is_training, dtype, aug_list=None):
    """Parses tfrecord data into list of tensors."""
    aug_ops_list = [
        augment.compose_augment_seq(aug_type, is_training=is_training)
        for aug_type in aug_list
    ]
    image, label, image_id = raw_record
    images = self.preprocess_image(image, dtype, aug_ops_list)
    label = tf.cast(tf.reshape(label, shape=[1]), dtype=tf.float32)
    return images + (label, image_id)

  def process_record_dataset(self,
                             dataset,
                             aug_list,
                             is_training,
                             batch_size,
                             shuffle_buffer,
                             dtype=tf.float32,
                             datasets_num_private_threads=None,
                             drop_remainder=False,
                             **kwargs):
    """Processes tfrecord dataset."""
    del kwargs

    # Define a specific size thread pool for tf.data operations.
    if datasets_num_private_threads:
      options = tf.data.Options()
      options.experimental_threading.private_threadpool_size = (
          datasets_num_private_threads)
      dataset = dataset.with_options(options)
      logging.info('datasets_num_private_threads: %s',
                   datasets_num_private_threads)

    if is_training:
      # Shuffle records before repeating to respect epoch boundaries.
      dataset = dataset.shuffle(
          buffer_size=shuffle_buffer, reshuffle_each_iteration=True)

    # Parse the raw records into images and labels.
    # pylint: disable=g-long-lambda
    dataset = dataset.map(
        lambda *args: self.parse_record_fn(
            args, is_training=is_training, dtype=dtype, aug_list=aug_list),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    if not is_training:
      num_batch = len([1 for _ in dataset.enumerate()])
    dataset = dataset.repeat()

    # Prefetch.
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset if is_training else [dataset, num_batch]

  def input_fn(self,
               is_training,
               batch_size,
               aug_list=None,
               dtype=tf.float32,
               num_cores=10,
               datasets_num_private_threads=None,
               input_context=None,
               training_dataset_cache=False,
               **kwargs):
    """Input function."""
    del kwargs

    # pylint: disable=assignment-from-no-return
    dataset = self.make_dataset(
        is_training=is_training,
        num_cores=num_cores,
        input_context=input_context)

    if is_training and training_dataset_cache:
      # Improve training performance when training data is in remote storage and
      # can fit into worker memory.
      dataset = dataset.cache()

    # aug_list should be a list of list of tuples.
    if not isinstance(aug_list, list):
      raise TypeError('augmentation list should be a list')
    if isinstance(aug_list, list):
      if not isinstance(aug_list[0], list):
        aug_list = [aug_list]

    return self.process_record_dataset(
        dataset=dataset,
        aug_list=aug_list,
        is_training=is_training,
        batch_size=batch_size,
        shuffle_buffer=1000,
        dtype=dtype,
        datasets_num_private_threads=datasets_num_private_threads,
        drop_remainder=True if is_training else False)

  def make_dataset(self, is_training, num_cores=10, input_context=None):
    pass


class ImageFromTFRecord(BasicImageProcess):
  """Image data loader from tfrecord."""

  def __init__(self, data_dir, input_shape=(256, 256, 3)):
    super(ImageFromTFRecord, self).__init__(input_shape=input_shape)
    self.data_dir = data_dir

  def wrap_int64(self, value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

  def wrap_bytes(self, value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

  def get_filenames(self, is_training):
    """Return filenames for dataset."""
    pass

  def parse_example_proto(self, example_serialized):
    """Parses example proto."""
    feature_map = {
        'image_id': tf.io.FixedLenFeature([], dtype=tf.int64, default_value=-1),
        'image': tf.io.FixedLenFeature([], dtype=tf.string, default_value=''),
        'label': tf.io.FixedLenFeature([], dtype=tf.int64, default_value=-1),
    }
    features = tf.io.parse_single_example(
        serialized=example_serialized, features=feature_map)
    label = tf.cast(features['label'], dtype=tf.int32)
    image_id = tf.cast(features['image_id'], dtype=tf.int32)
    return features['image'], label, image_id

  def raw_to_img(self, raw_record):
    image_buffer, label, image_id = self.parse_example_proto(
        example_serialized=raw_record)
    image = tf.io.decode_raw(image_buffer, tf.uint8)
    return image, label, image_id

  def make_dataset(self, is_training, num_cores=10, input_context=None):
    """Makes dataset."""
    # pylint: disable=assignment-from-no-return
    filenames = self.get_filenames(is_training)
    dataset = tf.data.Dataset.from_tensor_slices(filenames)

    if input_context:
      logging.info(
          'Sharding the dataset: input_pipeline_id=%d num_input_pipelines=%d',
          input_context.input_pipeline_id, input_context.num_input_pipelines)
      dataset = dataset.shard(input_context.num_input_pipelines,
                              input_context.input_pipeline_id)

    if is_training:
      # Shuffle the input files.
      dataset = dataset.shuffle(buffer_size=len(filenames))

    # Convert to individual records.
    dataset = dataset.interleave(
        tf.data.TFRecordDataset,
        cycle_length=num_cores,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # Decode.
    # pylint: disable=unnecessary-lambda
    dataset = dataset.map(
        lambda value: self.raw_to_img(value),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset


class ImageFromMemory(BasicImageProcess):
  """Image data loader from data tensor."""

  def __init__(self, data, input_shape=(32, 32, 3)):
    super(ImageFromMemory, self).__init__(input_shape=input_shape)
    self.data = data

  def make_dataset(self, is_training, num_cores=10, input_context=None):
    """Image data loader from data tensor."""

    dataset = tf.data.Dataset.from_tensor_slices(self.data)

    if input_context:
      logging.info(
          'Sharding the dataset: input_pipeline_id=%d num_input_pipelines=%d',
          input_context.input_pipeline_id, input_context.num_input_pipelines)
      dataset = dataset.shard(input_context.num_input_pipelines,
                              input_context.input_pipeline_id)
    if is_training:
      # Shuffle the input files.
      dataset = dataset.shuffle(buffer_size=len(self.data[0]))
    return dataset
