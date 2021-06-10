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
"""Metric library."""

import tensorflow as tf


class PerClassMeanAcc(tf.keras.metrics.Metric):
  """Per-class mean accuracy (recall)."""

  def __init__(self, num_class, name='per_class_mean_acc', **kwargs):
    super().__init__(name=name, **kwargs)
    self.count_correct = self.add_weight(
        name='count_correct', shape=[num_class], initializer='zeros')
    self.count = self.add_weight(
        name='count', shape=[num_class], initializer='zeros')
    self.num_class = num_class

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_true = tf.reshape(tf.cast(y_true, tf.int32), [-1])
    y_pred = tf.reshape(tf.cast(y_pred, tf.int32), [-1])

    count_class = tf.zeros(self.num_class)
    count_class_correct = tf.zeros(self.num_class)

    ones = tf.ones_like(y_true, tf.float32)
    correct = tf.cast(tf.equal(y_true, y_pred), tf.float32)

    count_class = tf.tensor_scatter_nd_add(count_class,
                                           tf.reshape(y_true, [-1, 1]), ones)
    count_class_correct = tf.tensor_scatter_nd_add(count_class_correct,
                                                   tf.reshape(y_true, [-1, 1]),
                                                   correct)

    self.count_correct.assign_add(count_class_correct)
    self.count.assign_add(count_class)

  def result(self):
    return tf.math.divide_no_nan(self.count_correct, self.count)

  def reset_states(self):
    self.count_correct.assign(tf.zeros(self.num_class))
    self.count.assign(tf.zeros(self.num_class))


class PerClassMeanPrecision(tf.keras.metrics.Metric):
  """Per-class mean precision."""

  def __init__(self, num_class, name='per_class_mean_precision', **kwargs):
    super().__init__(name=name, **kwargs)
    self.count_correct = self.add_weight(
        name='count_correct', shape=[num_class], initializer='zeros')
    self.count = self.add_weight(
        name='count', shape=[num_class], initializer='zeros')
    self.num_class = num_class

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_true = tf.reshape(tf.cast(y_true, tf.int32), [-1])
    y_pred = tf.reshape(tf.cast(y_pred, tf.int32), [-1])

    count_class = tf.zeros(self.num_class)
    count_class_correct = tf.zeros(self.num_class)

    ones = tf.ones_like(y_true, tf.float32)
    correct = tf.cast(tf.equal(y_true, y_pred), tf.float32)

    count_class = tf.tensor_scatter_nd_add(count_class,
                                           tf.reshape(y_pred, [-1, 1]), ones)
    count_class_correct = tf.tensor_scatter_nd_add(count_class_correct,
                                                   tf.reshape(y_pred, [-1, 1]),
                                                   correct)

    self.count_correct.assign_add(count_class_correct)
    self.count.assign_add(count_class)

  def result(self):
    return tf.math.divide_no_nan(self.count_correct, self.count)

  def reset_states(self):
    self.count_correct.assign(tf.zeros(self.num_class))
    self.count.assign(tf.zeros(self.num_class))


class ConfusionMatrix(tf.keras.metrics.Metric):
  """Confusion matrix."""

  def __init__(self, num_class, name='confusion_matrix', **kwargs):
    super().__init__(name=name, **kwargs)
    self.confusion_matrix = self.add_weight(
        name='confusion_matrix',
        shape=[num_class, num_class],
        initializer='zeros',
        dtype=tf.float32)
    self.num_class = num_class

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_true = tf.reshape(tf.cast(y_true, tf.int32), [-1])
    y_pred = tf.reshape(tf.cast(y_pred, tf.int32), [-1])

    zeros = tf.zeros([self.num_class, self.num_class], tf.float32)
    ones = tf.ones_like(y_pred, tf.float32)

    y_true = tf.expand_dims(y_true, axis=-1)
    y_pred = tf.expand_dims(y_pred, axis=-1)

    indices = tf.concat([y_true, y_pred], axis=-1)

    confusion_matrix_add = tf.tensor_scatter_nd_add(zeros, indices, ones)
    self.confusion_matrix.assign_add(confusion_matrix_add)

  def result(self):
    return self.confusion_matrix

  def reset_states(self):
    self.confusion_matrix.assign(tf.zeros([self.num_class, self.num_class]))
