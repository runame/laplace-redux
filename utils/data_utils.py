import numpy as np
import os
from PIL import Image

import torch
import torch.utils.data as data_utils
import torchvision.transforms.functional as TF
from torchvision import transforms, datasets

import utils.wilds_utils as wu


def get_in_distribution_data_loaders(args, device):
    """ load in-distribution datasets and return data loaders """

    if args.benchmark in ['R-MNIST', 'MNIST-OOD']:
        if args.benchmark == 'R-MNIST':
            no_loss_acc = False
            # here, id is the rotation angle
            ids = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]
        else:
            no_loss_acc = True
            # here, id is the name of the dataset
            ids = ['MNIST', 'EMNIST', 'FMNIST', 'KMNIST']
        train_loader, val_loader, in_test_loader = get_mnist_loaders(
            args.data_root,
            model_class=args.model,
            batch_size=args.batch_size,
            val_size=args.val_set_size,
            download=args.download,
            device=device)

    elif args.benchmark in ['R-FMNIST', 'FMNIST-OOD']:
        if args.benchmark == 'R-FMNIST':
            no_loss_acc = False
            # here, id is the rotation angle
            ids = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]
        else:
            no_loss_acc = True
            # here, id is the name of the dataset
            ids = ['FMNIST', 'EMNIST', 'MNIST', 'KMNIST']
        train_loader, val_loader, in_test_loader = get_fmnist_loaders(
            args.data_root,
            model_class=args.model,
            batch_size=args.batch_size,
            val_size=args.val_set_size,
            download=args.download,
            device=device)

    elif args.benchmark in ['R-FMNIST', 'FMNIST-OOD']:
        if args.benchmark == 'R-FMNIST':
            no_loss_acc = False
            # here, id is the rotation angle
            ids = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]
        else:
            no_loss_acc = True
            # here, id is the name of the dataset
            ids = ['FMNIST', 'EMNIST', 'MNIST', 'KMNIST']
        train_loader, val_loader, in_test_loader = get_fmnist_loaders(
            args.data_root,
            model_class=args.model,
            batch_size=args.batch_size,
            val_size=args.val_set_size,
            download=args.download,
            device=device)

    elif args.benchmark in ['CIFAR-10-C', 'CIFAR-10-OOD']:
        if args.benchmark == 'CIFAR-10-C':
            no_loss_acc = False
            # here, id is the corruption severity
            ids = [0, 1, 2, 3, 4, 5]
        else:
            no_loss_acc = True
            # here, id is the name of the OOD dataset
            ids = ['CIFAR-10', 'SVHN', 'LSUN', 'CIFAR-100']

        train_loader, val_loader, in_test_loader = get_cifar10_loaders(
            args.data_root,
            batch_size=args.batch_size,
            train_batch_size=args.batch_size,
            val_size=args.val_set_size,
            download=args.download,
            data_augmentation=not args.noda)

    elif args.benchmark == 'ImageNet-C':
        no_loss_acc = False
        # here, id is the corruption severity
        ids = [0, 1, 2, 3, 4, 5]
        train_loader, val_loader, in_test_loader = get_imagenet_loaders(
            args.data_root,
            batch_size=args.batch_size,
            train_batch_size=args.batch_size,
            val_size=args.val_set_size)

    elif 'WILDS' in args.benchmark:
        dataset = args.benchmark[6:]
        no_loss_acc = False
        ids = [f'{dataset}-id', f'{dataset}-ood']
        train_loader, val_loader, in_test_loader = wu.get_wilds_loaders(
            dataset, args.data_root, args.data_fraction, args.model_seed)

    return (train_loader, val_loader, in_test_loader), ids, no_loss_acc


def get_ood_test_loader(args, id):
    """ load out-of-distribution test data and return data loader """

    if args.benchmark == 'R-MNIST':
        _, test_loader = get_rotated_mnist_loaders(
            id, args.data_root,
            model_class=args.model,
            download=args.download)
    elif args.benchmark == 'R-FMNIST':
        _, test_loader = get_rotated_fmnist_loaders(
            id, args.data_root,
            model_class=args.model,
            download=args.download)
    elif args.benchmark == 'CIFAR-10-C':
        test_loader = load_corrupted_cifar10(
            id, data_dir=args.data_root,
            batch_size=args.batch_size,
            cuda=torch.cuda.is_available())
    elif args.benchmark == 'ImageNet-C':
        test_loader = load_corrupted_imagenet(
            id, data_dir=args.data_root,
            batch_size=args.batch_size,
            cuda=torch.cuda.is_available())
    elif args.benchmark == 'MNIST-OOD':
        _, test_loader = get_mnist_ood_loaders(
            id, data_path=args.data_root,
            batch_size=args.batch_size,
            download=args.download)
    elif args.benchmark == 'FMNIST-OOD':
        _, test_loader = get_mnist_ood_loaders(
            id, data_path=args.data_root,
            batch_size=args.batch_size,
            download=args.download)
    elif args.benchmark == 'CIFAR-10-OOD':
        _, test_loader = get_cifar10_ood_loaders(
            id, data_path=args.data_root,
            batch_size=args.batch_size,
            download=args.download)
    elif 'WILDS' in args.benchmark:
        dataset = args.benchmark[6:]
        test_loader = wu.get_wilds_ood_test_loader(
            dataset, args.data_root, args.data_fraction)

    return test_loader


def val_test_split(dataset, val_size=5000, batch_size=512, num_workers=5, pin_memory=False):
    # Split into val and test sets
    test_size = len(dataset) - val_size
    dataset_val, dataset_test = data_utils.random_split(
        dataset, (val_size, test_size), generator=torch.Generator().manual_seed(42)
    )
    val_loader = data_utils.DataLoader(dataset_val, batch_size=batch_size, shuffle=False,
                                       num_workers=num_workers, pin_memory=pin_memory)
    test_loader = data_utils.DataLoader(dataset_test, batch_size=batch_size, shuffle=False,
                                        num_workers=num_workers, pin_memory=pin_memory)
    return val_loader, test_loader


def get_cifar10_loaders(data_path, batch_size=512, val_size=2000,
                        train_batch_size=128, download=False, data_augmentation=True):
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    tforms = [transforms.ToTensor(),
              transforms.Normalize(mean, std)]
    tforms_test = transforms.Compose(tforms)
    if data_augmentation:
        tforms_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                                        transforms.RandomCrop(32, padding=4)]
                                        + tforms)
    else:
        tforms_train = tforms_test

    # Get datasets and data loaders
    train_set = datasets.CIFAR10(data_path, train=True, transform=tforms_train,
                                 download=download)
    # train_set = data_utils.Subset(train_set, range(500))
    val_test_set = datasets.CIFAR10(data_path, train=False, transform=tforms_test,
                                    download=download)

    train_loader = data_utils.DataLoader(train_set,
                                         batch_size=train_batch_size,
                                         shuffle=True)
    val_loader, test_loader = val_test_split(val_test_set,
                                             batch_size=batch_size,
                                             val_size=val_size)

    return train_loader, val_loader, test_loader


def get_imagenet_loaders(data_path, batch_size=128, val_size=2000,
                         train_batch_size=128, num_workers=5):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    tforms_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    tforms_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    data_path_train = os.path.join(data_path, 'ImageNet2012/train')
    data_path_val = os.path.join(data_path, 'ImageNet2012/val')

    # Get datasets and data loaders
    train_set = datasets.ImageFolder(data_path_train, transform=tforms_train)
    val_test_set = datasets.ImageFolder(data_path_val, transform=tforms_test)

    train_loader = data_utils.DataLoader(train_set,
                                         batch_size=train_batch_size,
                                         shuffle=True,
                                         num_workers=num_workers,
                                         pin_memory=True)
    val_loader, test_loader = val_test_split(val_test_set,
                                             batch_size=batch_size,
                                             val_size=val_size,
                                             num_workers=num_workers,
                                             pin_memory=True)

    return train_loader, val_loader, test_loader


def get_mnist_loaders(data_path, batch_size=512, model_class='LeNet',
                      train_batch_size=128, val_size=2000, download=False, device='cpu'):
    if model_class == "MLP":
        tforms = transforms.Compose([transforms.ToTensor(), ReshapeTransform((-1,))])
    else:
        tforms = transforms.ToTensor()

    train_set = datasets.MNIST(data_path, train=True, transform=tforms,
                               download=download)
    val_test_set = datasets.MNIST(data_path, train=False, transform=tforms,
                                  download=download)

    Xys = [train_set[i] for i in range(len(train_set))]
    Xs = torch.stack([e[0] for e in Xys]).to(device)
    ys = torch.stack([torch.tensor(e[1]) for e in Xys]).to(device)
    train_loader = FastTensorDataLoader(Xs, ys, batch_size=batch_size, shuffle=True)
    val_loader, test_loader = val_test_split(val_test_set,
                                             batch_size=batch_size,
                                             val_size=val_size)

    return train_loader, val_loader, test_loader


def get_fmnist_loaders(data_path, batch_size=512, model_class='LeNet',
                       train_batch_size=128, val_size=2000, download=False, device='cpu'):
    if model_class == "MLP":
        tforms = transforms.Compose([transforms.ToTensor(), ReshapeTransform((-1,))])
    else:
        tforms = transforms.ToTensor()

    train_set = datasets.FashionMNIST(data_path, train=True, transform=tforms,
                                      download=download)
    val_test_set = datasets.FashionMNIST(data_path, train=False, transform=tforms,
                                         download=download)

    Xys = [train_set[i] for i in range(len(train_set))]
    Xs = torch.stack([e[0] for e in Xys]).to(device)
    ys = torch.stack([torch.tensor(e[1]) for e in Xys]).to(device)
    train_loader = FastTensorDataLoader(Xs, ys, batch_size=batch_size, shuffle=True)
    val_loader, test_loader = val_test_split(val_test_set,
                                             batch_size=batch_size,
                                             val_size=val_size)

    return train_loader, val_loader, test_loader


def get_rotated_mnist_loaders(angle, data_path, model_class='LeNet', download=False):
    if model_class == "MLP":
        shift_tforms = transforms.Compose([RotationTransform(angle), transforms.ToTensor(),
                                           ReshapeTransform((-1,))])
    else:
        shift_tforms = transforms.Compose([RotationTransform(angle), transforms.ToTensor()])

    # Get rotated MNIST val/test sets and loaders
    rotated_mnist_val_test_set = datasets.MNIST(data_path, train=False,
                                                transform=shift_tforms,
                                                download=download)
    shift_val_loader, shift_test_loader = val_test_split(rotated_mnist_val_test_set,
                                                         val_size=2000)

    return shift_val_loader, shift_test_loader


def get_rotated_fmnist_loaders(angle, data_path, model_class='LeNet', download=False):
    if model_class == "MLP":
        shift_tforms = transforms.Compose([RotationTransform(angle), transforms.ToTensor(),
                                           ReshapeTransform((-1,))])
    else:
        shift_tforms = transforms.Compose([RotationTransform(angle), transforms.ToTensor()])

    # Get rotated FMNIST val/test sets and loaders
    rotated_fmnist_val_test_set = datasets.FashionMNIST(data_path, train=False,
                                                        transform=shift_tforms,
                                                        download=download)
    shift_val_loader, shift_test_loader = val_test_split(rotated_fmnist_val_test_set,
                                                         val_size=2000)

    return shift_val_loader, shift_test_loader


# https://discuss.pytorch.org/t/missing-reshape-in-torchvision/9452/6
class ReshapeTransform:
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, img):
        return torch.reshape(img, self.new_size)


class RotationTransform:
    """Rotate the given angle."""
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, x):
        return TF.rotate(x, self.angle)


def uniform_noise(dataset, delta=1, size=5000, batch_size=512):
    if dataset in ['MNIST', 'FMNIST', 'R-MNIST']:
        shape = (1, 28, 28)
    elif dataset in ['SVHN', 'CIFAR10', 'CIFAR100', 'CIFAR-10-C']:
        shape = (3, 32, 32)
    elif dataset in ['ImageNet', 'ImageNet-C']:
        shape = (3, 256, 256)

    # data = torch.rand((100*batch_size,) + shape)
    data = delta * torch.rand((size,) + shape)
    train = data_utils.TensorDataset(data, torch.zeros_like(data))
    loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
                                         shuffle=False, num_workers=1)
    return loader


class DatafeedImage(torch.utils.data.Dataset):
    def __init__(self, x_train, y_train, transform=None):
        self.x_train = x_train
        self.y_train = y_train
        self.transform = transform

    def __getitem__(self, index):
        img = self.x_train[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, self.y_train[index]

    def __len__(self):
        return len(self.x_train)


def load_corrupted_cifar10(severity, data_dir='data', batch_size=256, cuda=True,
                           workers=1):
    """ load corrupted CIFAR10 dataset """

    x_file = data_dir + '/CIFAR-10-C/CIFAR10_c%d.npy' % severity
    np_x = np.load(x_file)
    y_file = data_dir + '/CIFAR-10-C/CIFAR10_c_labels.npy'
    np_y = np.load(y_file).astype(np.int64)

    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    dataset = DatafeedImage(np_x, np_y, transform)
    dataset = data_utils.Subset(dataset, torch.randint(len(dataset), (10000,)))

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=cuda)

    return loader


def load_corrupted_imagenet(severity, data_dir='data', batch_size=128, cuda=True, workers=1):
    """ load corrupted ImageNet dataset """

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
    ])

    corruption_types = ['brightness', 'contrast', 'defocus_blur', 'elastic_transform', 'fog',
                        'frost', 'gaussian_blur', 'gaussian_noise', 'glass_blur', 'impulse_noise',
                        'jpeg_compression', 'motion_blur', 'pixelate', 'saturate', 'shot_noise',
                        'snow', 'spatter', 'speckle_noise', 'zoom_blur']

    dsets = list()
    for c in corruption_types:
        path = os.path.join(data_dir, 'ImageNet-C/' + c + '/' + str(severity))
        dsets.append(datasets.ImageFolder(path,
                                          transform=transform))
    dataset = data_utils.ConcatDataset(dsets)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=cuda)

    return loader


def get_mnist_ood_loaders(ood_dataset, data_path='./data', batch_size=512, download=False):
    '''Get out-of-distribution val/test sets and val/test loaders (in-distribution: MNIST/FMNIST)'''
    tforms = transforms.ToTensor()
    if ood_dataset == 'FMNIST':
        fmnist_val_test_set = datasets.FashionMNIST(data_path, train=False,
                                                    transform=tforms,
                                                    download=download)
        val_loader, test_loader = val_test_split(fmnist_val_test_set,
                                                 batch_size=batch_size,
                                                 val_size=0)
    elif ood_dataset == 'EMNIST':
        emnist_val_test_set = datasets.EMNIST(data_path, split='digits', train=False,
                                              transform=tforms,
                                              download=download)
        val_loader, test_loader = val_test_split(emnist_val_test_set,
                                                 batch_size=batch_size,
                                                 val_size=0)
    elif ood_dataset == 'KMNIST':
        kmnist_val_test_set = datasets.KMNIST(data_path, train=False,
                                              transform=tforms,
                                              download=download)
        val_loader, test_loader = val_test_split(kmnist_val_test_set,
                                                 batch_size=batch_size,
                                                 val_size=0)
    elif ood_dataset == 'MNIST':
        mnist_val_test_set = datasets.MNIST(data_path, train=False,
                                            transform=tforms,
                                            download=download)
        val_loader, test_loader = val_test_split(mnist_val_test_set,
                                                 batch_size=batch_size,
                                                 val_size=0)
    else:
        raise ValueError('Choose one out of FMNIST, EMNIST, MNIST, and KMNIST.')
    return val_loader, test_loader


def get_cifar10_ood_loaders(ood_dataset, data_path='./data', batch_size=512, download=False):
    '''Get out-of-distribution val/test sets and val/test loaders (in-distribution: CIFAR-10)'''
    if ood_dataset == 'SVHN':
        svhn_tforms = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.4376821, 0.4437697, 0.47280442),
                                                   (0.19803012, 0.20101562, 0.19703614))])
        svhn_val_test_set = datasets.SVHN(data_path, split='test',
                                          transform=svhn_tforms,
                                          download=download)
        val_loader, test_loader = val_test_split(svhn_val_test_set,
                                                 batch_size=batch_size,
                                                 val_size=0)
    elif ood_dataset == 'LSUN':
        lsun_tforms = transforms.Compose([transforms.Resize(size=(32, 32)),
                                          transforms.ToTensor()])
        lsun_test_set = datasets.LSUN(data_path, classes=['classroom_val'],  # classes='test'
                                      transform=lsun_tforms)
        val_loader = None
        test_loader = data_utils.DataLoader(lsun_test_set, batch_size=batch_size,
                                            shuffle=False)
    elif ood_dataset == 'CIFAR-100':
        cifar100_tforms = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                   (0.2023, 0.1994, 0.2010))])
        cifar100_val_test_set = datasets.CIFAR100(data_path, train=False,
                                                  transform=cifar100_tforms,
                                                  download=download)
        val_loader, test_loader = val_test_split(cifar100_val_test_set,
                                                 batch_size=batch_size,
                                                 val_size=0)
    else:
        raise ValueError('Choose one out of SVHN, LSUN, and CIFAR-100.')
    return val_loader, test_loader


class FastTensorDataLoader:
    """
    Source: https://github.com/hcarlens/pytorch-tabular/blob/master/fast_tensor_data_loader.py
    and https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """
    def __init__(self, *tensors, batch_size=32, shuffle=False):
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors
        self.dataset = tensors[0]

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches


class PermutedMnistGenerator():
    def __init__(self, data_path='./data', num_tasks=10, random_seed=0, download=False):
        self.data_path = data_path
        self.num_tasks = num_tasks
        self.random_seed = random_seed
        self.download = download
        self.out_dim = 10           # number of classes in the MNIST dataset
        self.in_dim = 784           # each image has 28x28 pixels
        self.task_id = 0            # initialize the current task id

    def next_task(self, batch_size=256, val_size=0):
        if self.task_id >= self.num_tasks:
            raise Exception('Number of tasks exceeded!')
        else:
            np.random.seed(self.task_id+self.random_seed)
            perm_inds = np.arange(self.in_dim)

            # First task is (unpermuted) MNIST, subsequent tasks are random permutations of pixels
            if self.task_id > 0:
                np.random.shuffle(perm_inds)

            # make image a tensor and permute pixel values
            tforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.view(-1)[perm_inds]),
            ])

            # load datasets
            train_set = datasets.MNIST(self.data_path, train=True,
                                       transform=tforms, download=self.download)
            val_test_set = datasets.MNIST(self.data_path, train=False,
                                          transform=tforms, download=self.download)

            # fast DataLoader for training
            Xys = [train_set[i] for i in range(len(train_set))]
            Xs = torch.stack([e[0] for e in Xys])
            ys = torch.stack([torch.tensor(e[1]) for e in Xys])
            train_loader = FastTensorDataLoader(Xs, ys, batch_size=batch_size, shuffle=True)
            val_loader, test_loader = val_test_split(val_test_set,
                                                     batch_size=batch_size,
                                                     val_size=val_size,
                                                     num_workers=0)

            # increment task counter
            self.task_id += 1

            if val_size > 0:
                return train_loader, val_loader, test_loader
            return train_loader, test_loader
