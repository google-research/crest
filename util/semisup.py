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
"""Trainer for semi-supervised learning."""

import functools
import os

from util.train import BaseTrain
import tensorflow as tf


class SemiSup(BaseTrain):
  """Semi-supervised learning."""

  def __init__(self, hparams):
    super().__init__(hparams)

    self.file_suffix = 'uratio{}_loss{}_div{}_th{:g}_T{:g}_wu{:g}'.format(
        self.unlab_ratio, self.unsup_loss_type, self.unsup_loss_divisor,
        self.threshold, self.temperature, self.weight_unsup)
    if self.do_distalign:
      self.file_suffix += '_dalign'
      if self.how_dalign:
        self.file_suffix += '_{}{}'.format(self.how_dalign, self.dalign_t)
    if self.reweight_labeled:
      self.file_suffix += '_rwl'
    self.set_ssl_hparams(hparams)

    self.metric_of_interest = 'acc.test.ema'
    self.list_of_metrics = [
        'loss.train', 'loss.xe', 'loss.xeu', 'loss.wd', 'acc.train',
        'acc.unlab', 'acc.test', 'acc.test.ema', 'acc.test.la',
        'acc.test.la.ema', 'per_class_acc.test', 'per_class_acc.test.ema',
        'per_class_acc.test.la', 'per_class_acc.test.la.ema', 'monitor.mask',
        'monitor.kl_data', 'monitor.kl_model'
    ]

    # Generate pseudo label.
    self.get_pseudo_target = functools.partial(
        self.pseudo_target_fn,
        threshold=self.threshold,
        temperature=self.temperature,
        do_distalign=self.do_distalign,
        how_dalign=self.how_dalign,
        stop_gradient=True)

  def set_hparams(self, hparams):
    """Sets algorithm-specific parameters."""
    self.unlab_ratio = hparams.unlab_ratio
    self.batch_size_unlab = self.batch_size * self.unlab_ratio
    self.unsup_loss_type = hparams.unsup_loss_type
    self.unsup_loss_divisor = hparams.unsup_loss_divisor
    self.threshold = hparams.threshold
    self.temperature = hparams.temperature
    self.weight_unsup = hparams.weight_unsup
    self.do_distalign = hparams.do_distalign

    # Set re-balancing parameters.
    self.how_dalign = hparams.how_dalign
    self.dalign_t = hparams.dalign_t
    self.reweight_labeled = hparams.reweight_labeled

  def set_ssl_hparams(self, hparams):
    pass

  def config(self, gen_idx, pseudo_label_list=None):
    """Defines an experiment configuration."""
    self.set_random_seed()
    # Data loader.
    dataloader, dl = self.get_dataloader(
        **{
            'class_im_ratio': self.class_im_ratio,
            'percent_labeled_per_class': self.percent_labeled,
            'update_mode': self.update_mode,
            'alpha': self.alpha,
            'pseudo_label_list': pseudo_label_list
        })
    self.train_loader = dataloader[0]
    self.test_loader = dataloader[-1]
    self.unlab_test_loader = dataloader[1]
    if self.strategy:
      self.train_loader = self.strategy.experimental_distribute_dataset(
          self.train_loader)
      self.test_loader[0] = self.strategy.experimental_distribute_dataset(
          self.test_loader[0])
      if self.unlab_test_loader:
        self.unlab_test_loader[
            0] = self.strategy.experimental_distribute_dataset(
                self.unlab_test_loader[0])
    self.num_class = dl.num_class
    self.gt_p_data = dl.gt_p_data
    self.db_name = dl.fname
    # Model architecture.
    self.model, self.model_ema = self.get_model(is_ema=True)
    self.ema_init()
    # Scheduler.
    self.scheduler, self.sched_name = self.get_scheduler(
        **{
            'step_size': self.sched_step_size,
            'gamma': self.sched_gamma,
            'min_rate': self.sched_min_rate,
            'level': self.sched_level
        })
    # Optimizer.
    self.optimizer, self.optim_name = self.get_optimizer(
        **{
            'momentum': self.momentum,
            'nesterov': self.nesterov,
            'clip_value': self.clip_value
        })
    # Re-initialize the step function.
    if hasattr(self, 'how_dalign') and self.how_dalign:
      if self.how_dalign == 'constant':
        current_dalign_t = self.dalign_t
      elif self.how_dalign == 'adaptive':
        cur = gen_idx / (self.num_generation - 1)
        current_dalign_t = (1.0 - cur) * 1.0 + cur * self.dalign_t
      else:
        raise NotImplementedError
      self.train_step_fn = self.get_train_step_fn(current_dalign_t)
      self.model_suffix = 'gen_{:02d}_t{:.2f}'.format(gen_idx, current_dalign_t)
    else:
      self.train_step_fn = self.get_train_step_fn()
      self.model_suffix = 'gen_{:02d}'.format(gen_idx)
    # Set file path.
    self.get_file_path_with_idx(gen_idx)

  def get_file_path_with_idx(self, gen_idx):
    """Gets file path."""
    if gen_idx == 0:
      self.get_file_path()
      self.file_path_prefix = self.file_path
    self.file_path = os.path.join(self.file_path_prefix, self.model_suffix)
    self.json_path = os.path.join(self.file_path, 'stats')

  def train_generations(self):
    """Trains a model with multiple sample generation steps."""
    pseudo_label_list = None
    for gen_idx in range(self.num_generation):
      self.strategy = self.setup_tf()
      with self.strategy.scope():
        self.config(gen_idx, pseudo_label_list)
        self.train(gen_idx)
        pseudo_label_list = self.eval()

  def train(self, gen_idx):
    """Trains a model."""
    self.train_begin()
    while self.get_epoch(is_tensor=False) < self.num_epoch:
      self.train_epoch_begin()
      self.train_epoch(desc='Train {:02d} Epoch '.format(gen_idx) +
                       '{epoch}/{num_epoch}')
      self.train_epoch_end()
    self.train_end()

  def get_train_step_fn(self, current_dalign_t=None):
    """Gets train step function."""

    @tf.function
    def step_fn(data):
      """Train step for Pseudo Label model.

      Args:
        data: A tuple of labeled and unlabeled data. Labeled data (data[0]) is
          an (data, label, index) tuple. Unlabeled data (data[1]) is an (data,
          label, index) tuple. Multiple augmented images of the same instance
          are available.
      """
      xl, yl = data[0][0], data[0][1]
      xu, yu, _ = data[1][0], data[1][-2], data[1][-1]
      replica_context = tf.distribute.get_replica_context()
      with tf.GradientTape() as tape:
        xc = tf.concat((xl, xu), axis=0)
        logits = self.model(xc, training=True)['logits']
        logits_l, logits_u = logits[:xl.shape[0]], logits[xl.shape[0]:]

        # Compute supervised loss.
        loss_xe = tf.keras.losses.sparse_categorical_crossentropy(
            yl, logits_l, from_logits=True)
        loss_xe = tf.divide(
            tf.reduce_sum(loss_xe),
            self.cross_replica_concat(loss_xe,
                                      replica_context=replica_context).shape[0])

        # Compute unsupervised loss.
        pseudo_target, pseudo_mask = self.get_pseudo_target(
            logits_u, current_dalign_t=current_dalign_t)
        loss_xeu = self.get_unsup_loss(
            pseudo_target, logits_u, mode=self.unsup_loss_type)
        loss_xeu = tf.reduce_sum(loss_xeu * pseudo_mask)
        loss_xeu = tf.divide(
            loss_xeu,
            self.get_unsup_loss_divisor(
                pseudo_mask,
                mode=self.unsup_loss_divisor,
                replica_context=replica_context))

        # Compute l2 weight decay loss.
        loss_wd = self.loss_wd(self.model.trainable_weights)
        loss = loss_xe + self.weight_decay * loss_wd
        loss += self.weight_unsup * loss_xeu
      grad = tape.gradient(loss, self.model.trainable_weights)
      self.optimizer.apply_gradients(zip(grad, self.model.trainable_weights))

      # EMA update.
      self.ema_update(ema_decay=self.ema_decay)

      # Metric monitor update.
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
          'monitor.mask':
              tf.reduce_mean(pseudo_mask),
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

  def eval(self):
    """Evaluates the model after each generation."""
    self.epoch = self.get_epoch(is_tensor=False)

    self.test_iterator = (iter(self.test_loader[0]), self.test_loader[1])
    test_scores, test_y = self.eval_dataset(self.test_iterator)
    self.metric_report(test_scores, test_y)

    pseudo_label_list = None
    if self.unlab_test_loader:
      self.unlab_iterator = (iter(self.unlab_test_loader[0]),
                             self.unlab_test_loader[1])
      unlab_scores, unlab_y = self.eval_dataset(self.unlab_iterator)
      self.metric_report(unlab_scores, unlab_y)
      pseudo_label_list = self.get_pseudo_label_list(unlab_scores)
    return pseudo_label_list

  def metric_report(self, scores, y_true):
    """Print metrics."""
    y_pred = tf.cast(tf.argmax(scores, axis=1), tf.int32)
    y_true = tf.cast(tf.squeeze(y_true), tf.int32)
    zeros = tf.zeros([self.num_class, self.num_class], tf.float32)
    ones = tf.ones_like(y_pred, tf.float32)
    indices = tf.stack([y_true, y_pred], axis=1)
    confusion_matrix = tf.tensor_scatter_nd_add(zeros, indices, ones)

    idx = tf.stack([tf.range(self.num_class), tf.range(self.num_class)], axis=1)
    correct = tf.gather_nd(confusion_matrix, idx)
    per_class_recall = correct / tf.reduce_sum(confusion_matrix, axis=1)
    per_class_precision = correct / tf.reduce_sum(confusion_matrix, axis=0)
    acc = tf.reduce_mean(per_class_recall)

    print('evaluting generation at epoch {:d}'.format(self.epoch))
    print('acc {:.3f}'.format(acc))
    print('per_class_recall {}'.format(per_class_recall))
    print('per_class_precision {}'.format(per_class_precision))
    print('confusion_matrix {}'.format(confusion_matrix))

  def get_pseudo_label_list(self, scores):
    """Generate per class pseudo label list."""
    y_pred = tf.argmax(scores, axis=1)
    y_score = tf.math.reduce_max(scores, axis=1)
    idx = tf.range(y_pred.shape)

    pseudo_label_list = []
    for class_idx in range(self.num_class):
      y_template = tf.ones_like(y_pred) * class_idx
      y_where = tf.squeeze(tf.where(y_pred == y_template))
      idx_gather = tf.reshape(tf.gather(idx, y_where), [-1])
      score_gather = tf.reshape(tf.gather(y_score, y_where), [-1])
      sort_idx = tf.argsort(score_gather, direction='DESCENDING')
      idx_gather = tf.gather(idx_gather, sort_idx)
      pseudo_label_list.append(idx_gather.numpy())

    return pseudo_label_list

  def pseudo_target_fn(self,
                       logits,
                       threshold=0.95,
                       temperature=0.0,
                       stop_gradient=True,
                       assert_nonempty=False,
                       do_distalign=False,
                       how_dalign=None,
                       current_dalign_t=1.0):
    """Gets pseudo target."""

    def _get_pseudo_target(logits, temperature):
      """Gets pseudo target from the list of logits.

      Args:
        logits: 2-D Tensor.
        temperature: Scalar for logit (or probability) scaling.

      Returns:
        pseudo_target: 2-D Tensor.
        pseudo_probs: 2-D Tensor.
      """
      if not tf.is_tensor(logits):
        return _get_pseudo_target_from_list(logits, temperature)

      # Get probability
      pseudo_probs = tf.nn.softmax(logits, axis=1)
      if do_distalign:
        if how_dalign:
          # Take temperature-scaled gt_p_data as target distribution.
          target_dist = tf.pow(self.gt_p_data, current_dalign_t)
          target_dist /= tf.reduce_sum(target_dist)
        else:
          # Take moving average of labeled set as target distribution.
          target_dist = self.p_data()
        pseudo_probs = (
            pseudo_probs * (1e-6 + target_dist) / (1e-6 + self.p_model()))
        pseudo_probs /= tf.reduce_sum(pseudo_probs, axis=1, keepdims=True)
      if temperature > 0.0:
        # Compute soft target.
        pseudo_target = tf.pow(pseudo_probs, tf.divide(1.0, temperature))
        pseudo_target = pseudo_target / tf.reduce_sum(
            pseudo_target, axis=1, keepdims=True)
      else:
        # Compute hard target.
        pseudo_target = tf.one_hot(
            tf.argmax(pseudo_probs, axis=1), logits.shape[1])
      return pseudo_target, pseudo_probs

    def _get_pseudo_target_from_list(logits, temperature):
      """Gets pseudo target from the list of logits.

      Tensors in the list should be of the same shape and the same row
      from different tensor are assumed to be from the same image with
      different augmentations.

      Args:
        logits: list of 2-D Tensor.
        temperature: Scalar for logit (or probability) scaling.

      Returns:
        Tuple of list of 2-D Tensors.
      """
      probs = [tf.nn.softmax(logit, axis=1) for logit in logits]
      pseudo_probs = tf.reduce_mean(tf.stack(probs, axis=0), axis=0)
      if do_distalign:
        if how_dalign:
          # Take temperature-scaled self.gt_p_data as target distribution.
          target_dist = tf.pow(self.gt_p_data, current_dalign_t)
          target_dist /= tf.reduce_sum(target_dist)
        else:
          # Take moving average of labeled set as target distribution.
          target_dist = self.p_data()
        pseudo_probs = (
            pseudo_probs * (1e-6 + target_dist) / (1e-6 + self.p_model()))
        pseudo_probs /= tf.reduce_sum(pseudo_probs, axis=1, keepdims=True)
      if temperature > 0.0:
        # Compute soft target.
        pseudo_target = tf.pow(pseudo_probs, tf.divide(1.0, temperature))
        pseudo_target = pseudo_target / tf.reduce_sum(
            pseudo_target, axis=1, keepdims=True)
      else:
        # Compute hard target.
        pseudo_target = tf.one_hot(
            tf.argmax(pseudo_probs, axis=1), pseudo_probs.shape[1])
      return (tf.concat([pseudo_target] * len(logits), axis=0),
              tf.concat([pseudo_probs] * len(logits), axis=0))

    pseudo_target, pseudo_probs = _get_pseudo_target(logits, temperature)
    threshold_max = tf.reduce_max(pseudo_probs)
    pseudo_mask = tf.cond(
        tf.logical_and(assert_nonempty, tf.less(threshold_max, threshold)),
        lambda: tf.reduce_max(pseudo_probs, axis=1) >= threshold_max,
        lambda: tf.reduce_max(pseudo_probs, axis=1) >= threshold)
    pseudo_mask = tf.cast(pseudo_mask, dtype=tf.float32)
    if stop_gradient:
      pseudo_target = tf.stop_gradient(pseudo_target)
      pseudo_probs = tf.stop_gradient(pseudo_probs)
    return pseudo_target, tf.stop_gradient(pseudo_mask)

  def get_unsup_loss_divisor(self,
                             pseudo_mask,
                             mode='full',
                             replica_context=None):
    """Gets divisor for unsupervised loss.

    Args:
      pseudo_mask: 1-D Tensor of 0 and 1.
      mode: 'full' (count all regardless of the confidence), 'quality' (count
        only examples that pass the threshold).
      replica_context: Context for multi-GPU or TPU training.

    Returns:
      Scalar.
    """
    pseudo_mask_concat = self.cross_replica_concat(pseudo_mask, replica_context)
    if mode == 'full':
      return tf.cast(pseudo_mask_concat.shape[0], dtype=tf.float32)
    elif mode == 'quality':
      return tf.maximum(1.0, tf.reduce_sum(pseudo_mask_concat))
    else:
      raise NotImplementedError

  def get_unsup_loss(self, targets, logits, mode='xe'):
    """Gets unsupervised loss.

    Args:
      targets: 1-D (label) or 2-D (probability) Tensor.
      logits: 2-D Tensor.
      mode: 'xe' for cross-entropy, 'mse' for mean squared error.

    Returns:
      1-D Tensor.
    """
    if mode == 'xe':
      return tf.keras.losses.categorical_crossentropy(
          targets, logits, from_logits=True)
    elif mode == 'mse':
      probs = tf.nn.softmax(logits, axis=1)
      return tf.reduce_mean(tf.square(targets - probs), axis=1)
    else:
      raise NotImplementedError
