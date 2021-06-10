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
"""Utilities to run model training and evaluation."""

from fixmatch import FixMatch
from mixmatch import MixMatch


def get_trainer(hparams):
  """Gets trainer."""
  if hparams.method.lower() == 'mixmatch':
    trainer = MixMatch(hparams)
  elif hparams.method.lower() == 'fixmatch':
    trainer = FixMatch(hparams)
  else:
    raise ValueError('method should be either mixmatch or fixmatch.')
  return trainer
