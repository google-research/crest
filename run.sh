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
#!/bin/bash
set -e
set -x

virtualenv -p python3 .
source ./bin/activate

pip install -r requirements.txt

# An example run for CIFAR10LT using CReST+
python -m train_and_eval_loop \
  --model_dir=/tmp/model \
  --method=fixmatch \
  --dataset=cifar10lt \
  --input_shape=32,32,3 \
  --class_im_ratio=0.01 \
  --percent_labeled=0.1 \
  --fold=1 \
  --num_epoch=64 \
  --unlab_ratio=7 \
  --temperature=0 \
  --num_generation=6 \
  --sched_level=1 \
  --dalign_t=0.5 \
  --how_dalign=adaptive \
  --do_distalign=True
