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
"""Train and eval loop of SSL algorithms."""

from absl import app
from absl import flags

import train_and_eval_lib

flags.DEFINE_string(name='root', default=None, help='Path to root data path.')
flags.DEFINE_string(name='method', default='fixmatch', help='method.')
# Dataset.
flags.DEFINE_string(name='dataset', default='cifar10', help='dataset.')
flags.DEFINE_string(name='source_data', default=None, help='source dataset(s).')
flags.DEFINE_string(name='target_data', default=None, help='target dataset.')
flags.DEFINE_integer(name='fold', default=1, help='data fold.')
flags.DEFINE_float('class_im_ratio', '0.1', 'class imbalance ratio.')
flags.DEFINE_float(
    name='percent_labeled',
    default=None,
    help='percent of labeled examples per class, only used by im settings.')
flags.DEFINE_integer(
    name='num_labeled', default=25, help='number of labeled data per class.')
flags.DEFINE_boolean(
    name='is_balanced',
    default=True,
    help='if True, labeled data from class-balanced distribution.')
flags.DEFINE_string(
    name='augment', default='dh', help='augmentation on labeled data.')
flags.DEFINE_string(
    name='weakaug', default='dh', help='weak augmentation on unlabeled data.')
flags.DEFINE_string(
    name='strongaug',
    default='dhrac',
    help='strong augmentation on unlabeled data.')
flags.DEFINE_integer(
    name='num_augment', default=1, help='number of augmentation per data.')
flags.DEFINE_integer(
    name='num_weakaug', default=1, help='number of weak augmentation per data.')
flags.DEFINE_integer(
    name='num_strongaug',
    default=1,
    help='number of strong augmentation per data.')
flags.DEFINE_string(
    name='input_shape', default='32,32,3', help='data input shape.')
# Network architecture.
flags.DEFINE_string(
    name='net_arch', default='WRN28', help='network architecture.')
flags.DEFINE_integer(
    name='net_width', default=2, help='network width. 16 filters when width=1.')
flags.DEFINE_string(
    name='activation', default='leaky_relu', help='network activation.')
flags.DEFINE_boolean(
    name='bn_sync',
    default=False,
    help='if True, use synchronized batch normalization.')
flags.DEFINE_string(
    name='head_arch', default='linear', help='head architecture.')
flags.DEFINE_string(
    name='head_mlp_dims', default='', help='head mlp dimension.')
# Optimization.
flags.DEFINE_integer(name='seed', default=0, help='random seed.')
flags.DEFINE_boolean(
    name='force_init',
    default=False,
    help='if False, continue training from existing checkpoint.')
flags.DEFINE_boolean(
    name='do_rollback',
    default=False,
    help='if True, do rollback when NaN occurs.')
flags.DEFINE_integer(name='num_workers', default=16, help='number of workers.')
flags.DEFINE_string(name='optim_type', default='sgd', help='optimizer type.')
flags.DEFINE_string(
    name='sched_type',
    default='halfcos',
    help='learning rate decay schedule type.')
flags.DEFINE_string(
    name='sched_freq',
    default='step',
    help='learning rate decay schedule frequency.')
flags.DEFINE_integer(
    name='sched_step_size',
    default=1,
    help='learning rate decay schedule step size.')
flags.DEFINE_float(
    name='sched_gamma',
    default=0.995,
    help='learning rate decay rate when sched_type is step.')
flags.DEFINE_float(
    name='sched_min_rate', default=0.0, help='minimum learning rate.')
flags.DEFINE_integer(
    name='sched_level',
    default=1,
    help='cosine learning rate decay schedule level.')
flags.DEFINE_float(name='learning_rate', default=0.03, help='learning rate.')
flags.DEFINE_float(
    name='weight_decay',
    default=0.0005,
    help='L2 weight regularization coefficient.')
flags.DEFINE_float(
    name='ema_decay',
    default=0.999,
    help='exponential moving average decay rate.')
flags.DEFINE_float(name='momentum', default=0.9, help='momentum.')
flags.DEFINE_boolean(
    name='nesterov', default=True, help='if True, nesterov momentum.')
flags.DEFINE_float(
    name='clip_value', default=0.0, help='gradient clip threshold.')
flags.DEFINE_integer(name='num_epoch', default=64, help='number of epoch.')
flags.DEFINE_integer(
    name='num_generation',
    default=15,
    help='number of pseudo-label generation.')
flags.DEFINE_integer(
    name='num_batch', default=1024, help='number of step (batch) per epoch.')
flags.DEFINE_integer(
    name='batch_size', default=64, help='size of labeled batch.')
# Monitoring and checkpoint.
flags.DEFINE_string(
    name='model_dir',
    default='',
    help='Path to output model directory where event and checkpoint files will be written.'
)
flags.DEFINE_string(name='ckpt_prefix', default='', help='checkpoint prefix.')
flags.DEFINE_string(name='file_path', default=None, help='file path.')
flags.DEFINE_string(name='file_suffix', default=None, help='file suffix.')
# SSL parameters
flags.DEFINE_integer(
    name='unlab_ratio', default=1, help='batch size ratio for unlabeled data.')
flags.DEFINE_string(
    name='unsup_loss_type',
    default='xe',
    help='type of unsupervised loss. xe or mse.')
flags.DEFINE_string(
    name='unsup_loss_divisor',
    default='full',
    help='divisor for unsupervised loss. full or quality.')
flags.DEFINE_float(
    name='weight_unsup', default=1.0, help='unsupervised loss coefficient.')
flags.DEFINE_float(name='threshold', default=0.95, help='confidence threshold.')
flags.DEFINE_float(
    name='temperature', default=1.0, help='temperature scaling for soft label.')
flags.DEFINE_float(name='mixup_beta', default=0.75, help='mixup parameter.')
# Distribution alignment.
flags.DEFINE_boolean(
    name='do_distalign', default=False, help='do distribution alignment.')
flags.DEFINE_string(
    name='how_dalign',
    default=None,
    help='how to use distribution alignment, `constant`, `adaptive`.')
flags.DEFINE_float(
    name='dalign_t',
    default=0.5,
    help='temperature for distribution alignment, paired use with hwo_dalign.')
flags.DEFINE_float(
    name='alpha',
    default=3.0,
    help='control sampling rate to update the labeled set.')
flags.DEFINE_boolean(
    name='reweight_labeled',
    default=False,
    help='whether reweight labeled data by inverse number.')
flags.DEFINE_string(
    name='update_mode',
    default='distribution',
    help='mode to update the labeled set, None, `distribution`, `all`')

FLAGS = flags.FLAGS


class HParams(dict):

  def __init__(self, *args, **kwargs):
    super(HParams, self).__init__(*args, **kwargs)
    self.__dict__ = self


def main(unused_argv):
  hparams = HParams({
      flag.name: flag.value for flag in FLAGS.get_flags_for_module('__main__')
  })
  # Starts training.
  trainer = train_and_eval_lib.get_trainer(hparams)
  trainer.train_generations()


if __name__ == '__main__':
  app.run(main)
