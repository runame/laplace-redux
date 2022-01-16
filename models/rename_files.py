import os
import sys


for fname in os.listdir(sys.argv[1]):
    seed = fname.split('_')[-1].split('.')[0]

    if 'MNIST_' in fname:
        os.rename(f'{sys.argv[1]}/{fname}', f'{sys.argv[1]}/lenet_mnist_{seed}_1.pt')
    elif 'CIFAR10_' in fname:
        os.rename(f'{sys.argv[1]}/{fname}', f'{sys.argv[1]}/wrn_16-4_cifar10_{seed}_1.pt')
