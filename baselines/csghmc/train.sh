#!/bin/bash

python baselines/csghmc/train.py --dataset MNIST --randseed 972394
python baselines/csghmc/train.py --dataset MNIST --randseed 12
python baselines/csghmc/train.py --dataset MNIST --randseed 523
python baselines/csghmc/train.py --dataset MNIST --randseed 13
python baselines/csghmc/train.py --dataset MNIST --randseed 6

python baselines/csghmc/train.py --dataset CIFAR10 --randseed 972394
python baselines/csghmc/train.py --dataset CIFAR10 --randseed 12
python baselines/csghmc/train.py --dataset CIFAR10 --randseed 523
python baselines/csghmc/train.py --dataset CIFAR10 --randseed 13
python baselines/csghmc/train.py --dataset CIFAR10 --randseed 6
