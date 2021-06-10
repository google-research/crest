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
"""Training utilities."""

from absl import logging

import tensorflow as tf
import tensorflow_probability as tfp


def setup_tf():
  logging.set_verbosity(logging.ERROR)
  physical_devices = tf.config.experimental.list_physical_devices('GPU')
  if not physical_devices:
    print('No GPUs are detected')
  for dev in physical_devices:
    tf.config.experimental.set_memory_growth(dev, True)
  return tf.distribute.MirroredStrategy()


def loss_wd(var_list):
  """Computes L2 weight decay loss."""
  return tf.add_n([tf.nn.l2_loss(v) for v in var_list if 'bn' not in v.name])


def clip_by_value(val, clip_value=1.0):
  """Clips by value."""

  def _clip_by_value(val, clip_value=1.0):
    return [(tf.clip_by_value(v, -clip_value, clip_value)) for v in val]

  return tf.cond(
      tf.greater(clip_value, 0), lambda: _clip_by_value(val, clip_value),
      lambda: val)


def kl_divergence(prob_a, prob_b, stop_gradient=True):
  """Computes KL divergence."""
  eps = 1e-6  # for numerical stability.
  prob_a = tf.clip_by_value(prob_a, eps, 1.0 - eps)
  prob_b = tf.clip_by_value(prob_b, eps, 1.0 - eps)
  loss = tf.reduce_sum(prob_a * tf.math.log(prob_a) -
                       prob_a * tf.math.log(prob_b))
  return tf.stop_gradient(loss) if stop_gradient else loss


def mixup(x1, l1, x2, l2, beta=0.75):
  """Mixup.

  Args:
    x1: N-D Tensor with the first dimension being the batch dimension.
    l1: 2-D Tensor for labels in one-hot format.
    x2: N-D Tensor with the first dimension being the batch dimension.
    l2: 2-D Tensor for labels in one-hot format.
    beta: Scalar for mixup coefficient.

  Returns:
    A tuple of N-D Tensor for mixed data and 2-D Tensor for mixed label.
  """
  # Shuffles x2 and l2 and subsample.
  multiple = tf.cast(
      tf.math.ceil(tf.divide(tf.shape(x1)[0],
                             tf.shape(x2)[0])), dtype=tf.int32)
  shuffle_index = tf.cast(
      tf.math.floormod(
          tf.random.shuffle(tf.range(tf.shape(x2)[0] * multiple)),
          tf.shape(x2)[0]),
      dtype=tf.int64)
  shuffle_index = tf.gather(shuffle_index, tf.range(tf.shape(x1)[0]))
  x2m = tf.gather(x2, shuffle_index)
  l2m = tf.gather(l2, shuffle_index)
  # Generates mixing values.
  mix = tf.cond(
      tf.greater(beta, 0),
      lambda: tfp.distributions.Beta(beta, beta).sample(tf.shape(x1)[0]),
      lambda: tf.ones(tf.shape(x1)[0]))
  mix = tf.reshape(tf.maximum(mix, 1.0 - mix), [tf.shape(x1)[0], 1, 1, 1])
  # Mixes input and output data.
  x1m = x1 * mix + x2m * (1 - mix)
  l1m = l1 * mix[:, :, 0, 0] + l2m * (1 - mix[:, :, 0, 0])
  return x1m, l1m


def cross_replica_concat(tensor, replica_context=None):
  """Reduce a concatenation of the `tensor` across TPU cores.

  Args:
    tensor: tensor to concatenate.
    replica_context: A `replica_context`. If not set, CPU execution is assumed.

  Returns:
    Tensor of the same rank as `tensor` with first dimension `num_replicas`
    times larger.
  """
  if replica_context is None or replica_context.num_replicas_in_sync <= 1:
    return tensor

  num_replicas = replica_context.num_replicas_in_sync

  with tf.name_scope('cross_replica_concat'):
    # This creates a tensor that is like the input tensor but has an added
    # replica dimension as the outermost dimension. On each replica it will
    # contain the local values and zeros for all other values that need to be
    # fetched from other replicas.
    ext_tensor = tf.scatter_nd(
        indices=[[replica_context.replica_id_in_sync_group]],
        updates=[tensor],
        shape=tf.concat([[num_replicas], tf.shape(tensor)], axis=0))

    # As every value is only present on one replica and 0 in all others,
    # adding them all together will result in the full tensor on all replicas.
    ext_tensor = replica_context.all_reduce(tf.distribute.ReduceOp.SUM,
                                            ext_tensor)

    # Flatten the replica dimension.
    # The first dimension size will be: tensor.shape[0] * num_replicas
    # Using [-1] trick to support also scalar input.
    return tf.reshape(ext_tensor, [-1] + ext_tensor.shape.as_list()[2:])


class PMovingAverage:
  """Class which accumilates moving average of distribution of labels."""

  def __init__(self, name, num_classes, buffer_size=128):
    # MEAN aggregation is used by DistributionStrategy to aggregate
    # variable updates across shards.
    self.ma = tf.Variable(
        tf.ones([buffer_size, num_classes]) / num_classes,
        trainable=False,
        name=name,
        aggregation=tf.VariableAggregation.MEAN)

  def __call__(self):
    v = tf.reduce_mean(self.ma, axis=0)
    return v / tf.reduce_sum(v)

  def update(self, entry):
    entry = tf.reduce_mean(entry, axis=0)
    self.ma.assign(tf.concat([self.ma[1:], [entry]], axis=0))


class PData:
  """Class which estimates probability of labels in data."""

  def __init__(self, num_classes):
    # MEAN aggregation is used by DistributionStrategy to aggregate
    # variable updates across shards.
    self.p_data = tf.Variable(
        tf.ones([num_classes]) / num_classes,
        trainable=False,
        name='p_data',
        aggregation=tf.VariableAggregation.MEAN)

  def __call__(self):
    return self.p_data / tf.reduce_sum(self.p_data)

  def update(self, entry, decay=0.999):
    entry = tf.reduce_mean(entry, axis=0)
    self.p_data.assign(self.p_data * decay + entry * (1 - decay))
