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
"""FixMatch implementation."""

from util.semisup import SemiSup
import tensorflow as tf


class FixMatch(SemiSup):
  """FixMatch."""

  def __init__(self, hparams):
    super().__init__(hparams)

    assert (self.num_augment == 1) and (
        len(self.augment)
        == 1), 'number of labeled data augmentation for {} should be 1'.format(
            self.__class__.__name__)
    assert (self.num_weakaug == 1) and (
        len(self.weakaug)
        == 1), 'number of weak augmentation for {} should be 1'.format(
            self.__class__.__name__)

    self.list_of_metrics += [
        'acc.unlab_mask', 'acc.strong', 'acc.strong_mask',
        'per_class_acc.train', 'per_class_monitor.model'
    ]

  def get_train_step_fn(self, current_dalign_t=None):
    """Train step."""

    @tf.function
    def step_fn(data):
      """Train step for FixMatch model.

      Args:
        data: Tuple of labeled and unlabeled data. Labeled data (data[0]) is an
          (images, label, index) tuple. Unlabeled data (data[1]) is an (images,
          label, index) tuple. Multiple augmented images of the same instance
          are available.
      """
      xl, yl = data[0][0], data[0][-2]
      xu, yu, yi = data[1][:self.num_weakaug], data[1][-2], data[1][-1]
      xs = data[1][self.num_weakaug:-2]
      num_aug = len(xs)  # number of strong augmentations
      xu = tf.concat(xu, axis=0)
      xs = tf.concat(xs, axis=0)
      xu = tf.concat([xu for _ in range(num_aug)], axis=0)
      yu = tf.concat([yu for _ in range(num_aug)], axis=0)
      yi = tf.concat([yi for _ in range(num_aug)], axis=0)
      replica_context = tf.distribute.get_replica_context()
      if self.reweight_labeled:
        reweight_labeled_weights = 1 / (1e-6 + self.p_data())
        reweight_labeled_weights /= tf.reduce_sum(reweight_labeled_weights)
        reweight_labeled_weights *= self.num_class
      with tf.GradientTape() as tape:
        xc = tf.concat((xl, xu, xs), axis=0)
        logits = self.model(xc, training=True)['logits']
        logits_l = logits[:xl.shape[0]]
        logits_u, logits_s = tf.split(logits[xl.shape[0]:], 2)

        # Compute supervised loss.
        loss_xe = tf.keras.losses.sparse_categorical_crossentropy(
            yl, logits_l, from_logits=True)
        if self.reweight_labeled:
          loss_xe *= tf.gather(reweight_labeled_weights,
                               tf.cast(yl[:, 0], tf.int32))
        loss_xe = tf.divide(
            tf.reduce_sum(loss_xe),
            self.cross_replica_concat(loss_xe,
                                      replica_context=replica_context).shape[0])

        # Compute unsupervised loss.
        pseudo_target, pseudo_mask = self.get_pseudo_target(
            logits_u, current_dalign_t=current_dalign_t)
        loss_xeu = self.get_unsup_loss(
            pseudo_target, logits_s, mode=self.unsup_loss_type)
        loss_xeu = tf.reduce_sum(loss_xeu * pseudo_mask)
        loss_xeu = tf.divide(
            loss_xeu,
            self.get_unsup_loss_divisor(
                pseudo_mask,
                mode=self.unsup_loss_divisor,
                replica_context=replica_context))

        # Compute l2 weight decay loss.
        loss_wd = self.loss_wd(self.model.trainable_weights)

        # Compute total loss.
        loss = loss_xe + self.weight_decay * loss_wd
        loss += self.weight_unsup * loss_xeu
      grad = tape.gradient(loss, self.model.trainable_weights)
      grad = self.clip_by_value(grad, clip_value=self.clip_value)
      self.optimizer.apply_gradients(zip(grad, self.model.trainable_weights))

      # EMA update.
      self.ema_update(ema_decay=self.ema_decay)

      # Metric monitor update.
      pseudo_index = tf.where(tf.equal(pseudo_mask, 1))[:, 0]
      self.metric_update({
          'loss.train':
              loss * self.strategy.num_replicas_in_sync,
          'loss.xe':
              loss_xe * self.strategy.num_replicas_in_sync,
          'loss.xeu':
              loss_xeu * self.strategy.num_replicas_in_sync,
          'loss.wd':
              loss_wd * self.strategy.num_replicas_in_sync,
          'acc.train': (yl, tf.argmax(logits_l, axis=1)),
          'acc.unlab': (yu, tf.argmax(logits_u, axis=1)),
          'acc.strong': (yu, tf.argmax(logits_s, axis=1)),
          'acc.unlab_mask':
              (tf.gather(yu, pseudo_index),
               tf.argmax(tf.gather(logits_u, pseudo_index), axis=1)),
          'acc.strong_mask':
              (tf.gather(yu, pseudo_index),
               tf.argmax(tf.gather(logits_s, pseudo_index), axis=1)),
          'per_class_acc.train': (yl, tf.argmax(logits_l, axis=1)),
          'per_class_monitor.model':
              self.p_model(),
          'monitor.mask':
              tf.reduce_mean(pseudo_mask),
          'monitor.kl_data':
              self.kl_divergence(
                  prob_a=tf.ones([self.num_class]) / self.num_class,
                  prob_b=self.p_data()),
          'monitor.kl_model':
              self.kl_divergence(
                  prob_a=tf.ones([self.num_class]) / self.num_class,
                  prob_b=self.p_model()),
      })

      # Update model and data distributions.
      self.p_model.update(tf.stop_gradient(tf.nn.softmax(logits_u)))
      self.p_data.update(
          tf.one_hot(tf.cast(tf.squeeze(yl), dtype=tf.int32), self.num_class))

    return step_fn
