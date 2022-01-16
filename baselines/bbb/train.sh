#!/bin/bash

python baselines/bbb/train.py --dataset MNIST --randseed 972394
python baselines/bbb/train.py --dataset MNIST --randseed 12
python baselines/bbb/train.py --dataset MNIST --randseed 523
python baselines/bbb/train.py --dataset MNIST --randseed 13
python baselines/bbb/train.py --dataset MNIST --randseed 6

python baselines/bbb/train.py --dataset CIFAR10 --randseed 972394
python baselines/bbb/train.py --dataset CIFAR10 --randseed 12
python baselines/bbb/train.py --dataset CIFAR10 --randseed 523
python baselines/bbb/train.py --dataset CIFAR10 --randseed 13
python baselines/bbb/train.py --dataset CIFAR10 --randseed 6
