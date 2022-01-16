import os, sys
import os.path as o
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), '../..')))

import torch
import torch.nn.functional as F
from torch import optim
from baselines.bbb.models.lenet import LeNetBBB
from baselines.bbb.models.wrn import WideResNetBBB
from utils import data_utils, test
from tqdm import tqdm, trange
import numpy as np
import argparse
import pickle
import math
from torch.cuda import amp


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='MNIST', choices=['MNIST', 'CIFAR10'])
parser.add_argument('--estimator', default='flipout', choices=['reparam', 'flipout'])
parser.add_argument('--var0', type=float, default=1, help='Gaussian prior variance. If None, it will be computed to emulate weight decay')
parser.add_argument('--tau', type=float, default=0.1, help='Tempering parameter for the KL-term')
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

if args.var0 is None:
    args.var0 = 1/(5e-4*len(train_loader.dataset))

if args.dataset == 'MNIST':
    model = LeNetBBB(num_classes, var0=args.var0, estimator=args.estimator)
    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
    arch_name = 'lenet'
    dir_name = 'lenet_mnist'
else:
    model = WideResNetBBB(16, 4, num_classes, var0=args.var0, estimator=args.estimator)
    opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0, nesterov=True)
    arch_name = 'wrn_16-4'
    dir_name = 'wrn_16-4_cifar10'

print(f'Num. params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

model.cuda()
model.train()

n_epochs = 100
pbar = trange(n_epochs)
## T_max is the max iterations: n_epochs x n_batches_per_epoch
scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs*len(train_loader))

## For automatic-mixed-precision
scaler = amp.GradScaler()

# Timing stuff
timing_start = torch.cuda.Event(enable_timing=True)
timing_end = torch.cuda.Event(enable_timing=True)
torch.cuda.synchronize()
timing_start.record()

for epoch in pbar:
    train_loss = 0
    num_batches = len(train_loader)
    num_data = len(train_loader.dataset)

    for batch_idx, (x, y) in enumerate(train_loader):
        model.train()
        opt.zero_grad()

        m = len(x)  # Batch size
        x, y = x.cuda(non_blocking=True), y.long().cuda(non_blocking=True)

        with amp.autocast():
            out, kl = model(x)
            # Scaled negative-ELBO with 1 MC sample
            # See Graves 2011 as to why the KL-term is scaled that way and notice that we use mean instead of sum; tau is the tempering parameter
            loss = F.cross_entropy(out.squeeze(), y) + args.tau/num_data*kl

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        scheduler.step()

        train_loss = 0.9*train_loss + 0.1*loss.item()

    model.eval()
    pred = test.predict_vb(test_loader, model, n_samples=1).cpu().numpy()
    acc_val = np.mean(np.argmax(pred, 1) == targets)*100
    mmc_val = pred.max(-1).mean()*100

    pbar.set_description(
        f'[Epoch: {epoch+1}; ELBO: {train_loss:.3f}; acc: {acc_val:.1f}; mmc: {mmc_val:.1f}]'
    )

# Timing stuff
timing_end.record()
torch.cuda.synchronize()
timing = timing_start.elapsed_time(timing_end)/1000
np.save(f'results/timings_train/bbb-{args.estimator}_{args.dataset.lower()}_{args.randseed}', timing)

path = f'./models/bbb/{args.estimator}/{dir_name}'

if not os.path.exists(path):
    os.makedirs(path)

save_name = f'{path}/{arch_name}_{args.dataset.lower()}_{args.randseed}_1'
torch.save(model.state_dict(), save_name)

## Try loading and testing
model.load_state_dict(torch.load(save_name))
model.eval()

print()

## In-distribution
py_in = test.predict_vb(test_loader, model, n_samples=20).cpu().numpy()
acc_in = np.mean(np.argmax(py_in, 1) == targets)*100
print(f'Accuracy: {acc_in:.1f}')
