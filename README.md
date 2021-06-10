# CReST in Tensorflow 2

Code for the paper: "[CReST: A Class-Rebalancing Self-Training Framework for Imbalanced Semi-Supervised Learning](https://arxiv.org/abs/2102.09559)" by Chen Wei, Kihyuk Sohn, Clayton Mellina, Alan Yuille and Fan Yang.

-   **This is not an officially supported Google product.**

## Install dependencies

```bash
sudo apt install python3-dev python3-virtualenv python3-tk imagemagick
virtualenv -p python3 --system-site-packages env3
. env3/bin/activate
pip install -r requirements.txt
```

-   The code has been tested on Ubuntu 18.04 with CUDA 10.2.

## Environment setting

```bash
. env3/bin/activate
export ML_DATA=/path/to/your/data
export ML_DIR=/path/to/your/code
export RESULT=/path/to/your/result
export PYTHONPATH=$PYTHONPATH:$ML_DIR
```

## Datasets

Download or generate the datasets as follows:

-   CIFAR10 and CIFAR100: Follow the [steps](https://github.com/google-research/fixmatch/blob/master/README.md#install-datasets) to download and generate balanced CIFAR10 and CIFAR100 datasets. Put it under `${ML_DATA}/cifar`, for example, `${ML_DATA}/cifar/cifar10-test.tfrecord`.
-  Long-tailed CIFAR10 and CIFAR100: Follow the [steps](https://github.com/richardaecn/class-balanced-loss#datasets) to download the datasets prepared by Cui et al. Put it under `${ML_DATA}/cifar-lt`, for example, `${ML_DATA}/cifar-lt/cifar-10-data-im-0.1`.


## Running experiment on Long-tailed CIFAR10, CIFAR100

Run [MixMatch](mixmatch.py) ([paper](https://arxiv.org/abs/1905.02249)) and [FixMatch](fixmatch.py) ([paper](https://arxiv.org/abs/2001.07685)):

-   Specify method to run via `--method`. It can be `fixmatch` or `mixmatch`.
-   Specify dataset via `--dataset`. It can be `cifar10lt` or `cifar100lt`.
-   Specify the class imbalanced ratio, i.e., the number of training samples from the most minority class over that from the most majority class, via `--class_im_ratio`.
-   Specify the percentage of labeled data via `--percent_labeled`.
-   Specify the number of generations for self-training via `--num_generation`.
-   Specify whether to use distribution alignment via `--do_distalign`.
-   Specify the initial distribution alignment temperature via `--dalign_t`.
-   Specify how distribution alignment is applied via `--how_dalign`. It can be `constant` or `adaptive`.

    ```bash
    python -m train_and_eval_loop \
      --model_dir=/tmp/model \
      --method=fixmatch \
      --dataset=cifar10lt \
      --input_shape=32,32,3 \
      --class_im_ratio=0.01 \
      --percent_labeled=0.1 \
      --fold=1 \
      --num_epoch=64 \
      --num_generation=6 \
      --sched_level=1 \
      --dalign_t=0.5 \
      --how_dalign=adaptive \
      --do_distalign=True
    ```

## Results

The code reproduces main results of the paper. For all settings and methods, we run experiments on 5 different folds and report the mean and standard deviations. Note that the numbers may not exactly match those from the papers as there are extra randomness coming from the training.

**Results on Long-tailed CIFAR10 with 10% labeled data (Table 1 in the paper).**
|          | gamma=50    | gamma=100   | gamma=200   |
|----------|-------------|-------------|-------------|
| FixMatch | 79.4 (0.98) | 66.2 (0.83) | 59.9 (0.44) |
| CReST    | 83.7 (0.40) | 75.4 (1.62) | 63.9 (0.67) |
| CReST+   | 84.5 (0.41) | 77.7 (1.22) | 67.5 (1.36) |


## Training with Multiple GPUs

-   Simply set `CUDA_VISIBLE_DEVICES=0,1,2,3` or any number of GPUs.
-   Make sure that `batch size` is divisible by the number of GPUs.

## Augmentation

-   One can concatenate different augmentation shortkeys to compose an
    augmentation sequence.
    -   `d`: default augmentation, resize and shift.
    -   `h`: horizontal flip.
    -   `ra`: random augment with all augmentation ops.
    -   `rc`: random augment with color augmentation ops only.
    -   `rg`: random augment with geometric augmentation ops only.
    -   `c`: cutout.
    -   For example, `dhrac` applies shift, flip, random augment with all ops,
        followed by cutout.

## Citing this work

```bibtex
@article{wei2021crest,
    title={CReST: A Class-Rebalancing Self-Training Framework for Imbalanced Semi-Supervised Learning},
    author={Chen Wei and Kihyuk Sohn and Clayton Mellina and Alan Yuille and Fan Yang},
    journal={arXiv preprint arXiv:2102.09559},
    year={2021},
}
```
