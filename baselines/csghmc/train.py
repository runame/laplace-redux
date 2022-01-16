import os, sys
import os.path as o
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), '../..')))

import torch
import torch.nn.functional as F
from torch import optim
from baselines.vanilla.models.lenet import LeNet
from baselines.vanilla.models.wrn import WideResNet
from baselines.csghmc.csghmc import CSGHMCTrainer
from utils import data_utils, test
from tqdm import tqdm, trange
import numpy as np
import argparse
import pickle
import math
import copy


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='MNIST', choices=['MNIST', 'CIFAR10'])
parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--n_cycles', type=int, default=4)
parser.add_argument('--n_samples_per_cycle', type=int, default=3)
parser.add_argument('--initial_lr', type=float, default=0.1)
parser.add_argument('--randseed', type=int, default=123)
args = parser.parse_args()

np.random.seed(args.randseed)
torch.manual_seed(args.randseed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# Just symlink your dataset folder into your home directory like so
# No need to change this code---this way it's more consistent
data_path = os.path.expanduser('~/Datasets')

if args.dataset == 'MNIST':
    train_loader, val_loader, test_loader = data_utils.get_mnist_loaders(data_path)
elif args.dataset == 'CIFAR10':
    train_loader, val_loader, test_loader = data_utils.get_cifar10_loaders(data_path)

targets = torch.cat([y for x, y in test_loader], dim=0).numpy()
num_classes = 100 if args.dataset == 'CIFAR100' else 10

if args.dataset in ['MNIST', 'FMNIST']:
    model = LeNet(num_classes)
    arch_name = 'lenet'
    dir_name = 'lenet_mnist'
else:
    model = WideResNet(16, 4, num_classes, dropRate=0)
    arch_name = 'wrn_16-4'
    dir_name = 'wrn_16-4_cifar10'

print(f'Num. params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

model.cuda()
model.train()

batch_size = 128
data_size = len(train_loader.dataset)
num_batch = data_size/batch_size + 1
epoch_per_cycle = args.n_epochs // args.n_cycles
pbar = trange(args.n_epochs)
total_iters = args.n_epochs * num_batch
weight_decay = 5e-4

# Timing stuff
timing_start = torch.cuda.Event(enable_timing=True)
timing_end = torch.cuda.Event(enable_timing=True)
torch.cuda.synchronize()
timing_start.record()

trainer = CSGHMCTrainer(
    model, args.n_cycles, args.n_samples_per_cycle, args.n_epochs, args.initial_lr,
    num_batch, total_iters, data_size, weight_decay
)
samples = []

for epoch in pbar:
    train_loss = 0
    num_batches = len(train_loader)
    num_data = len(train_loader.dataset)

    for batch_idx, (x, y) in enumerate(train_loader):
        trainer.model.train()
        trainer.model.zero_grad()

        x, y = x.cuda(non_blocking=True), y.long().cuda(non_blocking=True)

        out = trainer.model(x)
        loss = F.cross_entropy(out, y)
        loss.backward()

        # The meat of the CSGMCMC method
        lr = trainer.adjust_lr(epoch, batch_idx)
        trainer.update_params(epoch)

        train_loss = 0.9*train_loss + 0.1*loss.item()

    # Save the last n_samples_per_cycle iterates of a cycle
    if (epoch % epoch_per_cycle) + 1 > epoch_per_cycle - args.n_samples_per_cycle:
        samples.append(copy.deepcopy(trainer.model.state_dict()))

    model.eval()
    pred = test.predict(test_loader, model).cpu().numpy()
    acc_val = np.mean(np.argmax(pred, 1) == targets)*100
    mmc_val = pred.max(-1).mean()*100

    pbar.set_description(
        f'[Epoch: {epoch+1}; loss: {train_loss:.3f}; acc: {acc_val:.1f}; mmc: {mmc_val:.1f}]'
    )

# Timing stuff
timing_end.record()
torch.cuda.synchronize()
timing = timing_start.elapsed_time(timing_end)/1000
np.save(f'results/timings_train/csghmc_{args.dataset.lower()}_{args.randseed}', timing)

path = f'./models/csghmc/{dir_name}'

if not os.path.exists(path):
    os.makedirs(path)

save_name = f'{path}/{arch_name}_{args.dataset.lower()}_{args.randseed}_1'
torch.save(samples, save_name)

## Try loading and testing
samples_state_dicts = torch.load(save_name)
models = []

for state_dict in samples_state_dicts:
    if args.dataset in ['MNIST', 'FMNIST']:
        _model = LeNet(num_classes)
    else:
        _model = WideResNet(16, 4, num_classes, dropRate=0)

    _model.load_state_dict(state_dict)
    models.append(_model.cuda().eval())

print()

py_in = test.predict_ensemble(test_loader, models).cpu().numpy()
acc_in = np.mean(np.argmax(py_in, 1) == targets)*100
print(f'Accuracy: {acc_in:.1f}')
