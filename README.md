# Laplace Redux - Effortless Bayesian Deep Learning

This repository contains the code to run the experiments for the paper [Laplace Redux - Effortless Bayesian Deep Learning](https://arxiv.org/abs/2106.14806) (NeurIPS 2021), using our library [laplace](https://github.com/AlexImmer/Laplace/).

## Requirements

After cloning the repository and creating a new virtual environment, install the package including all requirements with:
```
pip install .
```
For the BBB baseline, please follow the instructions in the [corresponding README](baselines/bbb).

For running the WILDS experiments, please follow the instructions for installing the WILDS library and the required dependencies in the [WILDS GitHub repository](https://github.com/p-lambda/wilds). Our experiments also require the `transformers` library (as mentioned in the WILDS GitHub repo under the section `Installation/Default models`). Our experiments were run and tested with version 1.1.0 of the WILDS library.


## Uncertainty Quantification Experiments (Sections 4.2 and 4.3)

The script `uq.py` runs the distribution shift (rotated (F)MNIST, corrupted CIFAR-10) and OOD ((F)MNIST and CIFAR-10 as in-distribution) experiments reported in Section 4.2, as well as the experiments on the WILDS benchmark reported in Section 4.3.
It expects pre-trained models, which can be downloaded [here](https://nc.mlcloud.uni-tuebingen.de/index.php/s/8fgF2y8SDkSwcsX); they should be placed in the [models](./models/) directory. Due to the large filesize the SWAG models are not included. Please contact us if you are interested in obtaining them.

To more conveniently run the experiments with the same parameters as we used in the paper, we provide some dedicated config files for the results with the Laplace approximation (`{x/y}` highlights options `x` and `y`); note that you might want to change the `download` flag or the `data_root` in the config file:
```
python uq.py --benchmark {R-MNIST/MNIST-OOD} --config configs/post_hoc_laplace/mnist_{default/bestood}.yaml
python uq.py --benchmark {CIFAR-10-C/CIFAR-10-OOD} --config configs/post_hoc_laplace/cifar10_{default/bestood}.yaml
```
The config files with `*_default` contains the default library setting of the Laplace approximation (`LA` in the paper) and `*_bestood` the setting which performs best on OOD data (`LA*` in the paper).

For running the baselines, take a look at the commands in `run_uq_baslines.sh`.


## Continual Learning Experiments (Section 4.4)

Run
```
python continual_learning.py
```
to reproduce the `LA-KFAC` result and run
```
python continual_learning.py --hessian_structure diag
```
to reproduce the `LA-DIAG` result of the continual learning experiment in Section 4.4.


## Training Baselines

In order to train the baselines, please note the following:

* Symlink your dataset dir to your `~/Datasets`, e.g. `ln -s /your/dataset/dir ~/Datasets`.
* Always run the training scripts from the project's root directory, e.g. `python baselines/bbb/train.py`.
