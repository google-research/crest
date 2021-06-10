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
"""The base trainer."""

import itertools
import json
import os
import random
import shutil
import time

from data import cifar
from model import resnet as model
from third_party import ema
import class_imbalanced_ssl.util.metric as util_metric
import class_imbalanced_ssl.util.scheduler as util_scheduler
import class_imbalanced_ssl.util.util as util
import numpy as np
import tensorflow as tf
from tqdm import trange


class BaseTrain(object):
  """Base model trainer."""

  def __init__(self, hparams):
    # data
    self.dataset = hparams.dataset
    self.augment = hparams.augment.split(',')
    self.weakaug = hparams.weakaug.split(',')
    self.strongaug = hparams.strongaug.split(',')
    self.num_augment = hparams.num_augment
    self.num_weakaug = hparams.num_weakaug
    self.num_strongaug = hparams.num_strongaug
    self.augment_list = [(self.augment, self.num_augment),
                         (self.weakaug, self.num_weakaug),
                         (self.strongaug, self.num_strongaug)]
    self.num_strongaug = hparams.num_strongaug
    self.input_shape = tuple([int(s) for s in hparams.input_shape.split(',')])
    self.fold = hparams.fold
    self.num_labeled = hparams.num_labeled
    self.percent_labeled = hparams.percent_labeled
    self.is_balanced = hparams.is_balanced
    self.class_im_ratio = hparams.class_im_ratio
    self.num_generation = hparams.num_generation
    # network
    self.net_arch = hparams.net_arch
    self.net_width = hparams.net_width
    self.activation = hparams.activation
    self.bn_sync = hparams.bn_sync
    self.head_arch = hparams.head_arch
    self.head_mlp_dims = tuple([
        int(d) for d in hparams.head_mlp_dims.split(',')
    ]) if hparams.head_mlp_dims not in ['', None] else None
    # optimizer
    self.seed = hparams.seed
    self.force_init = hparams.force_init
    self.num_workers = hparams.num_workers
    self.optim_type = hparams.optim_type
    self.sched_type = hparams.sched_type
    self.sched_freq = hparams.sched_freq
    self.sched_step_size = hparams.sched_step_size
    self.sched_gamma = hparams.sched_gamma
    self.sched_min_rate = hparams.sched_min_rate
    self.sched_level = hparams.sched_level
    self.learning_rate = hparams.learning_rate
    self.weight_decay = hparams.weight_decay
    self.ema_decay = hparams.ema_decay
    self.momentum = hparams.momentum
    self.nesterov = hparams.nesterov
    self.clip_value = hparams.clip_value
    self.num_epoch = hparams.num_epoch
    self.num_batch = hparams.num_batch
    self.batch_size = hparams.batch_size
    # checkpoint
    self.model_dir = None if hparams.model_dir == 'None' else hparams.model_dir
    self.ckpt_prefix = os.path.join(self.model_dir, hparams.ckpt_prefix)
    self.file_path = hparams.file_path
    self.file_suffix = hparams.file_suffix
    # additional hparams
    self.update_mode = hparams.update_mode
    self.alpha = hparams.alpha
    self.set_hparams(hparams=hparams)
    self.hparams = hparams

  def setup_tf(self):
    return util.setup_tf()

  def set_hparams(self, hparams):
    del hparams
    self.metric_of_interest = 'acc.test'

  def set_random_seed(self):
    seed = self.seed
    if seed > 0:
      random.seed(seed)
      np.random.seed(seed)
      tf.random.set_seed(seed)

  def ema_init(self):
    ema.assign_ema_vars_from_initial_values(self.model_ema.variables,
                                            self.model.variables)

  def ema_update(self, ema_decay=0.99):
    ema.update_ema_variables(self.model_ema.variables, self.model.variables,
                             tf.cast(ema_decay, dtype=tf.float32))

  def config(self, **kwargs):
    """Defines an experiment configuration."""
    del kwargs

    self.set_random_seed()
    # Data loader.
    dataloader, dl = self.get_dataloader()
    self.train_loader = dataloader[0]
    self.test_loader = dataloader[-1]
    if self.strategy:
      self.train_loader = self.strategy.experimental_distribute_dataset(
          self.train_loader)
      self.test_loader[0] = self.strategy.experimental_distribute_dataset(
          self.test_loader[0])
    self.num_class = dl.num_class
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
    # Define train step function.
    self.train_step_fn = self.get_train_step_fn()
    # Set file path.
    self.get_file_path()

  def get_dataset(self, **kwargs):
    """Gets dataset."""
    if self.dataset == 'cifar10':
      return cifar.CIFAR10()
    elif self.dataset == 'cifar100':
      return cifar.CIFAR100()
    elif self.dataset == 'cifar10lt':
      return cifar.CIFAR10LT(kwargs.get('class_im_ratio', 1))
    elif self.dataset == 'cifar100lt':
      return cifar.CIFAR100LT(kwargs.get('class_im_ratio', 1))
    else:
      raise ValueError('Dataset not supported.')

  def get_dataloader(self, **kwargs):
    """Gets dataloader."""
    dl = self.get_dataset(**kwargs)
    data_loader = dl.load_dataset(
        fold=self.fold,
        num_labeled_per_class=self.num_labeled,
        is_balanced=self.is_balanced,
        input_shape=self.input_shape,
        augment=self.augment_list,
        batch_size=self.batch_size,
        batch_size_unlab=self.batch_size_unlab if hasattr(
            self, 'batch_size_unlab') else 0,
        num_workers=self.num_workers,
        strategy=self.strategy,
        **kwargs)

    return data_loader, dl

  def get_model(self, is_ema=True):
    """Gets model."""

    net = model.__dict__[self.net_arch](
        head=self.head_arch,
        head_mlp=self.head_mlp_dims,
        input_shape=self.input_shape,
        num_class=self.num_class,
        width=self.net_width,
        activation=self.activation,
        bn_sync=self.bn_sync)
    net.summary()
    if is_ema:
      net_ema = model.__dict__[self.net_arch](
          head=self.head_arch,
          head_mlp=self.head_mlp_dims,
          input_shape=self.input_shape,
          num_class=self.num_class,
          width=self.net_width,
          activation=self.activation,
          bn_sync=self.bn_sync)
      return (net, net_ema)
    return (net,)

  def get_optimizer(self, **kwargs):
    """Gets optimizer."""
    if self.optim_type == 'sgd':
      momentum = kwargs['momentum'] if 'momentum' in kwargs else 0.9
      nesterov = kwargs['nesterov'] if 'nesterov' in kwargs else False
      clip_value = kwargs['clip_value'] if 'clip_value' in kwargs else None
      args = {'momentum': momentum, 'nesterov': nesterov}
      optimizer = tf.keras.optimizers.SGD(learning_rate=self.scheduler, **args)
      name = 'sgd_lr{}_mom{}'.format(self.learning_rate, momentum)
      if clip_value is not None and clip_value > 0.0:
        name = '{}_clip{:g}'.format(name, clip_value)
      if nesterov:
        name += '_nesterov'
    elif self.optim_type == 'adam':
      optimizer = tf.keras.optimizers.Adam(
          learning_rate=self.scheduler, amsgrad=True)
      name = 'adam_lr{}'.format(self.learning_rate)
    else:
      raise NotImplementedError
    return optimizer, name

  def get_scheduler(self, **kwargs):
    """Gets scheduler."""

    scheduler = util_scheduler.CustomLearningRateSchedule(
        step_per_epoch=1 if self.sched_freq == 'step' else self.num_batch,
        base_lr=self.learning_rate,
        max_step=self.num_epoch * self.num_batch,
        mode=self.sched_type,
        **kwargs)
    return scheduler, scheduler.name

  def get_file_path(self, **kwargs):
    """Gets file path."""
    del kwargs
    if not self.file_path:
      self.file_path = os.path.join(
          self.db_name, self.model.name,
          '{}_{}_{}_wd{:g}_ema{:g}_epoch{}_nb{}_bs{}'.format(
              self.__class__.__name__, self.optim_name, self.sched_name,
              self.weight_decay, self.ema_decay, self.num_epoch, self.num_batch,
              self.batch_size))
    self.file_path = os.path.join(self.ckpt_prefix, self.file_path)
    if self.file_suffix is not None and self.file_suffix:
      self.file_path = '{}_{}'.format(self.file_path, self.file_suffix)
    self.json_path = os.path.join(self.file_path, 'stats')

  def get_checkpoint(self):
    """Loads from checkpoint."""
    self.checkpoint.restore(self.manager.latest_checkpoint)
    self.checkpoint_ema.restore(self.manager_ema.latest_checkpoint)
    self.epoch = self.get_epoch(is_tensor=False)

  def get_epoch(self, is_tensor=True):
    """Returns current training epoch."""
    epoch = tf.math.floordiv(self.optimizer.iterations, self.num_batch)
    return epoch if is_tensor else epoch.numpy()

  def get_step(self, is_tensor=True):
    """Returns current training step."""
    step = self.optimizer.iterations
    return step if is_tensor else step.numpy()

  def train_generations(self):
    """Trains a model without a sample generation step."""
    self.strategy = self.setup_tf()
    with self.strategy.scope():
      self.config()
      self.train()

  def train(self, **kwargs):
    """Trains a model."""
    del kwargs
    self.train_begin()
    while self.get_epoch(is_tensor=False) < self.num_epoch:
      self.train_epoch_begin()
      self.train_epoch(desc='Epoch {epoch}/{num_epoch}')
      self.train_epoch_end()
    self.train_end()

  def train_begin(self):
    """Calls at the beginning of model training.

    Define metrics, model checkpoints, their managers, tensorboard, and load
    from the existing checkpoint if not force initialization from scratch.

    """
    self.metrics = {}
    self.metrics.update({
        key: tf.keras.metrics.Mean()
        for key in self.list_of_metrics
        if key.startswith(('loss', 'monitor'))
    })
    self.metrics.update({
        key: tf.keras.metrics.Accuracy()
        for key in self.list_of_metrics
        if key.startswith('acc')
    })
    self.metrics.update({
        key: util_metric.PerClassMeanAcc(self.num_class)
        for key in self.list_of_metrics
        if key.startswith('per_class_acc')
    })
    self.metrics.update({
        key: tf.keras.metrics.MeanTensor()
        for key in self.list_of_metrics
        if key.startswith('per_class_monitor')
    })
    self.monitor = {
        'learning_rate': None,
        'step_per_second': None,
    }
    if self.force_init:
      shutil.rmtree(self.file_path, ignore_errors=True)
    # Checkpoint
    self.checkpoint = tf.train.Checkpoint(
        optimizer=self.optimizer, model=self.model)
    self.manager = tf.train.CheckpointManager(
        checkpoint=self.checkpoint,
        directory=os.path.join(self.file_path, 'raw'),
        max_to_keep=4)
    # Checkpoint for EMA model
    self.checkpoint_ema = tf.train.Checkpoint(model=self.model_ema)
    self.manager_ema = tf.train.CheckpointManager(
        checkpoint=self.checkpoint_ema,
        directory=os.path.join(self.file_path, 'ema'),
        max_to_keep=4)
    self.tensorboard_dir = os.path.join(self.file_path, 'tb')
    self.summary_writer = tf.summary.create_file_writer(
        logdir=self.tensorboard_dir)
    self.train_iterator = iter(self.train_loader)
    self.test_iterator = (iter(self.test_loader[0]), self.test_loader[1])
    # Initialize data and model distributions.
    self.p_data = util.PData(self.num_class)
    self.p_model = util.PMovingAverage('p_model', self.num_class)
    # Load from checkpoint.
    self.get_checkpoint()

  def train_end(self):
    """Calls at the end of model training.

    Save summary statistics in json format and in command line.
    """
    self.summary_writer.close()
    if not tf.io.gfile.isdir(self.json_path):
      tf.io.gfile.makedirs(self.json_path)
    # Write hparams
    with tf.io.gfile.GFile(os.path.join(self.json_path, 'hparams.json'),
                           'w') as outfile:
      json.dump(self.hparams, outfile, sort_keys=True, indent=4)
    logdir = self.tensorboard_dir
    event_files = list(tf.io.gfile.glob(os.path.join(logdir, '*')))
    event_files.sort(key=lambda filename: tf.io.gfile.stat(filename).mtime_nsec)
    event_dict = {
        key: [] for key in self.metrics.keys() if not key.startswith('monitor')
    }
    for event_file in event_files:
      for event in tf.compat.v1.train.summary_iterator(event_file):
        for v in event.summary.value:
          if v.tag.replace('/', '.') in event_dict:
            event_dict[v.tag.replace('/', '.')].append(
                tf.make_ndarray(v.tensor).tolist())
    # Print stats of last 50 epochs in json format
    num_epoch_to_save = 50
    event_dict = {
        key: event_dict[key][-num_epoch_to_save:] for key in event_dict
    }
    for key in event_dict:
      dict_to_write = {
          'median (last%02d)' % x: np.median(event_dict[key][-x:])
          for x in [1, 10, 20, num_epoch_to_save]
      }
      dict_to_write.update({'last%02d' % (num_epoch_to_save,): event_dict[key]})
      with tf.io.gfile.GFile(os.path.join(self.json_path, key + '.json'),
                             'w') as outfile:
        json.dump(dict_to_write, outfile, sort_keys=True, indent=4)
      if key == self.metric_of_interest:
        summary_dict = {key: dict_to_write}
        with tf.io.gfile.GFile(
            os.path.join(self.json_path, 'summary.json'), 'w') as outfile:
          json.dump(summary_dict, outfile, sort_keys=True, indent=4)
    # Print basic information
    print('---------------------------------------------------------------')
    print('Train is done. Below are file path and basic test stats')
    print(self.file_path)
    del summary_dict[self.metric_of_interest]['last%02d' % (num_epoch_to_save,)]
    print(json.dumps(summary_dict, sort_keys=True, indent=4))
    print('---------------------------------------------------------------')

  def train_epoch(self, desc):
    """Trains a model for one epoch."""
    time_init = time.time()
    for _ in trange(
        self.num_batch,
        leave=False,
        desc=desc.format(epoch=self.epoch + 1, num_epoch=self.num_epoch)):
      self.train_step(self.train_iterator)
    self.monitor['step_per_second'] = self.num_batch / (time.time() - time_init)

  def train_epoch_begin(self):
    """Calls at the beginning of train epoch."""
    for _, metric in self.metrics.items():
      metric.reset_states()
    self.epoch = self.get_epoch(is_tensor=False)
    self.monitor['learning_rate'] = self.optimizer.learning_rate(
        self.optimizer.iterations).numpy()

  def train_epoch_end(self):
    """Calls at the end of train epoch."""
    if hasattr(self, 'val_iterator'):
      self.eval_epoch(self.val_iterator, is_val=True)
    for is_ema, use_logit_adjustment in itertools.product([False, True],
                                                          [False, True]):
      self.eval_epoch(self.test_iterator, False, is_ema, use_logit_adjustment)
    self.monitor_progress()
    self.manager.save()
    self.manager_ema.save()

  def get_train_step_fn(self):
    """Gets train step function."""

    @tf.function
    def step_fn(data):
      """Train step for supervised model.

      Args:
        data: A tuple of (data, label, index).
      """
      x, y = data[0], data[1]
      replica_context = tf.distribute.get_replica_context()
      with tf.GradientTape() as tape:
        logits = self.model(x, training=True)['logits']
        loss_xe = tf.keras.losses.sparse_categorical_crossentropy(
            y, logits, from_logits=True)
        loss_xe = tf.divide(
            tf.reduce_sum(loss_xe),
            self.cross_replica_concat(loss_xe,
                                      replica_context=replica_context).shape[0])
        loss_wd = self.loss_wd(self.model.trainable_weights)
        loss = loss_xe + self.weight_decay * loss_wd
      grad = tape.gradient(loss, self.model.trainable_weights)
      self.optimizer.apply_gradients(zip(grad, self.model.trainable_weights))

      # EMA update.
      self.ema_update(ema_decay=self.ema_decay)

      # Metric monitor update.
      self.metric_update({
          'loss.train': loss * self.strategy.num_replicas_in_sync,
          'loss.xe': loss_xe,
          'loss.wd': loss_wd,
          'acc.train': (y, tf.argmax(logits, axis=1)),
      })
    return step_fn

  def train_step(self, iterator):
    """Trains a model for one step."""
    self.strategy.run(self.train_step_fn, args=(next(iterator),))

  def eval_dataset(self,
                   dataset,
                   is_ema=True,
                   use_logit_adjustment=False,
                   accuracy_metric=None,
                   per_class_accuracy_metric=None,
                   desc=''):
    """Evaluates a model on the dataset."""

    inference = self.model_ema if is_ema else self.model
    iterator, num_batch = dataset[0], dataset[1]
    scores_list, y_list = [], []
    for _ in trange(num_batch, leave=False, desc=desc):
      scores, y = self.eval_step(
          iterator,
          inference,
          use_logit_adjustment,
          accuracy_metric,
          per_class_accuracy_metric,
      )
      scores_list.append(scores)
      y_list.append(y)
    score = tf.concat(scores_list, axis=0)
    y = tf.concat(y_list, axis=0)
    return score, y

  def eval_epoch(self,
                 dataset,
                 is_val=True,
                 is_ema=False,
                 use_logit_adjustment=False):
    """Evaluates a model at epoch end."""

    prefix = '.ema' if is_ema else ''
    prefix = '.la' + prefix if use_logit_adjustment else prefix
    accuracy = self.metrics['acc.val' + prefix if is_val else 'acc.test' +
                            prefix]
    per_class_accuracy = self.metrics[
        'per_class_acc.val' + prefix if is_val else 'per_class_acc.test' +
        prefix]
    desc = 'Epoch (eval) %d/%d' % (self.epoch + 1, self.num_epoch)
    self.eval_dataset(dataset, is_ema, use_logit_adjustment, accuracy,
                      per_class_accuracy, desc)

  @tf.function
  def eval_step(self,
                iterator,
                inference,
                use_logit_adjustment=False,
                accuracy_metric=None,
                per_class_accuracy_metric=None):
    """Evaluates a model for one step."""

    def step_fn(data):
      x, y = data[0], data[1]
      logits = inference(x, training=False)['logits']
      if use_logit_adjustment:
        logits = logits - tf.expand_dims(tf.math.log(self.p_data()), axis=0)
      scores = tf.nn.softmax(logits, axis=1)
      if accuracy_metric:
        accuracy_metric.update_state(y, tf.argmax(logits, axis=1))
      if per_class_accuracy_metric:
        per_class_accuracy_metric.update_state(y, tf.argmax(logits, axis=1))
      return scores, y

    return self.strategy.run(step_fn, args=(next(iterator),))

  def monitor_progress(self):
    """Updates monitoring variables."""

    # Print on tensorboard.
    with self.summary_writer.as_default():
      vis_step = (self.epoch + 1) * self.num_batch
      for key, metric in self.metrics.items():
        if key.startswith('per_class'):
          result_tensor = metric.result()
          for i in range(self.num_class):
            tf.summary.scalar(
                '{}_{}'.format(key.replace('.', '/', 1), i),
                result_tensor[i],
                step=vis_step)
        else:
          tf.summary.scalar(
              key.replace('.', '/', 1), metric.result(), step=vis_step)
      tf.summary.scalar(
          'monitor/step_per_second',
          self.monitor['step_per_second'],
          step=vis_step)
      tf.summary.scalar(
          'monitor/lr', self.monitor['learning_rate'], step=vis_step)

    # Print on the command line.
    template = ('Epoch {epoch:4d}/{num_epoch:4d}\tstep(sec): '
                '{step_per_second:.3f}\t')
    template += ('Loss: {loss:.3f}\tacc(tr): {acc_tr:.3f}\tacc(ts): '
                 '{acc_ts:.3f}\tema(ts): {ema_ts:.3f}')
    print(
        template.format(
            epoch=self.epoch + 1,
            num_epoch=self.num_epoch,
            step_per_second=self.monitor['step_per_second'],
            loss=self.metrics['loss.train'].result(),
            acc_tr=self.metrics['acc.train'].result(),
            acc_ts=self.metrics['acc.test'].result(),
            ema_ts=self.metrics['acc.test.ema'].result()))

  def metric_update(self, metrics):
    """Updates metrics."""
    for key in metrics:
      if isinstance(metrics[key], tuple):
        self.metrics[key].update_state(*metrics[key])
      else:
        self.metrics[key].update_state(metrics[key])

  def loss_wd(self, var_list):
    """Computes L2 weight decay loss."""
    return tf.divide(util.loss_wd(var_list), self.strategy.num_replicas_in_sync)

  def kl_divergence(self, prob_a, prob_b, stop_gradient=True):
    """Computes KL divergence."""
    return util.kl_divergence(prob_a, prob_b, stop_gradient)

  def clip_by_value(self, val, clip_value=1.0):
    """Clips by value."""
    return util.clip_by_value(val, clip_value)

  def cross_replica_concat(self, tensor, replica_context=None):
    """Reduce a concatenation of the `tensor` across TPU cores."""
    return util.cross_replica_concat(tensor, replica_context)

  def get_mixed_data(self, x1, l1, x2, l2, beta=0.75, replica_context=None):
    """Gets Mixed data.

    Args:
      x1: N-D Tensor with the first dimension being the batch dimension.
      l1: 2-D Tensor for labels in one-hot format.
      x2: N-D Tensor with the first dimension being the batch dimension.
      l2: 2-D Tensor for labels in one-hot format.
      beta: Scalar for mixup coefficient.
      replica_context: Context for multi-GPU or TPU training.

    Returns:
      Tuple of N-D Tensor for mixed data and 2-D Tensor for mixed label.
    """
    x2_concat = self.cross_replica_concat(x2, replica_context)
    l2_concat = self.cross_replica_concat(l2, replica_context)
    # pylint: disable=g-long-lambda
    return tf.cond(
        tf.reduce_all((tf.shape(x1)[0] > 0, tf.shape(x2_concat)[0] > 0),
                      tf.greater(beta, 0)),
        lambda: util.mixup(x1, l1, x2_concat, l2_concat, beta), lambda:
        (x1, l1))
