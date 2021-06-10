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
"""MixMatch implementation."""

from util.semisup import SemiSup
import tensorflow as tf


class MixMatch(SemiSup):
  """MixMatch."""

  def __init__(self, hparams):
    super().__init__(hparams)

    assert (self.num_augment == 1) and (
        len(self.augment)
        == 1), 'number of labeled data augmentation for {} should be 1'.format(
            self.__class__.__name__)
    assert (self.num_strongaug == 0) and (
        len(self.strongaug)
        == 1), 'number of strong augmentation for {} should be 0'.format(
            self.__class__.__name__)

    self.list_of_metrics += ['per_class_acc.train', 'per_class_monitor.model']

  def set_ssl_hparams(self, hparams):
    self.mixup_beta = hparams.mixup_beta
    self.mixup_prob = hparams.mixup_prob
    self.file_suffix += '_beta{:g}'.format(self.mixup_beta)

  def get_train_step_fn(self, current_dalign_t=None):
    """Train step."""

    @tf.function
    def step_fn(data):
      """Train step for MixMatch model.

      Args:
        data: Tuple of labeled and unlabeled data. Labeled data (data[0]) is an
          (images, label, index) tuple. Unlabeled data (data[1]) is an (images,
          label, index) tuple. Multiple augmented images of the same instance
          are available.
      """
      xl, yl = data[0][0], data[0][-2]
      xu, yu, _ = data[1][:self.num_weakaug], data[1][-2], data[1][-1]
      num_aug = len(xu)
      xu = tf.concat(xu, axis=0)
      replica_context = tf.distribute.get_replica_context()
      if self.reweight_labeled:
        reweight_labeled_weights = 1 / (1e-6 + self.p_data())
        reweight_labeled_weights /= tf.reduce_sum(reweight_labeled_weights)
        reweight_labeled_weights *= self.num_class
      with tf.GradientTape() as tape:
        # MixUp
        logits_u = self.model(xu, training=True)['logits']
        pseudo_target, pseudo_mask = self.get_pseudo_target(
            tf.split(logits_u, num_aug), current_dalign_t=current_dalign_t)
        xmix, ymix = self.get_mixed_data(
            x1=tf.concat([xl, xu], axis=0),
            l1=tf.concat([
                tf.one_hot(tf.cast(yl[:, 0], dtype=tf.int32), self.num_class),
                pseudo_target
            ],
                         axis=0),
            x2=tf.concat([xl, xu], axis=0),
            l2=tf.concat([
                tf.one_hot(tf.cast(yl[:, 0], dtype=tf.int32), self.num_class),
                pseudo_target
            ],
                         axis=0),
            beta=self.mixup_beta,
            replica_context=replica_context)
        logits = self.model(xmix, training=True)['logits']
        logits_l, logits_m = logits[:xl.shape[0]], logits[xl.shape[0]:]
        labels_l, labels_m = ymix[:xl.shape[0]], ymix[xl.shape[0]:]

        # Compute supervised loss.
        loss_xe = tf.keras.losses.categorical_crossentropy(
            labels_l, logits_l, from_logits=True)
        if self.reweight_labeled:
          loss_xe *= tf.gather(reweight_labeled_weights,
                               tf.cast(yl[:, 0], tf.int32))
        loss_xe = tf.divide(
            tf.reduce_sum(loss_xe),
            self.cross_replica_concat(loss_xe,
                                      replica_context=replica_context).shape[0])

        # Compute unsupervised loss.
        loss_xeu = self.get_unsup_loss(
            labels_m, logits_m, mode=self.unsup_loss_type)
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
        loss = loss + self.weight_unsup * loss_xeu
      grad = tape.gradient(loss, self.model.trainable_weights)
      self.optimizer.apply_gradients(zip(grad, self.model.trainable_weights))

      # EMA update.
      self.ema_update(ema_decay=self.ema_decay)

      # Metric monitor update.
      self.metric_update({
          'loss.train': loss * self.strategy.num_replicas_in_sync,
          'loss.xe': loss_xe * self.strategy.num_replicas_in_sync,
          'loss.xeu': loss_xeu * self.strategy.num_replicas_in_sync,
          'loss.wd': loss_wd * self.strategy.num_replicas_in_sync,
          'acc.train': (yl, tf.argmax(logits_l, axis=1)),
          'acc.unlab': (yu, tf.argmax(tf.split(logits_u, num_aug)[0], axis=1)),
          'per_class_acc.train': (yl, tf.argmax(logits_l, axis=1)),
          'per_class_monitor.model': self.p_model(),
          'monitor.mask': tf.reduce_mean(pseudo_mask),
          'monitor.kl_data':
              self.kl_divergence(
                  prob_a=tf.ones([self.num_class]) / self.num_class,
                  prob_b=self.p_data()),
          'monitor.kl_model':
              self.kl_divergence(
                  prob_a=tf.ones([self.num_class]) / self.num_class,
                  prob_b=self.p_model())
      })

      # Update model and data distributions.
      self.p_model.update(tf.stop_gradient(tf.nn.softmax(logits_u)))
      self.p_data.update(
          tf.one_hot(tf.cast(tf.squeeze(yl), dtype=tf.int32), self.num_class))

    return step_fn
