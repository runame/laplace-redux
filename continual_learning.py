import os
import argparse
import yaml
from copy import deepcopy

import numpy as np
import torch
from torch.nn.utils import parameters_to_vector
from torch.optim.lr_scheduler import CosineAnnealingLR

from laplace import Laplace
from laplace.curvature import AsdlGGN, AsdlEF

import utils.data_utils as du
import utils.utils as util


def main(args):
    # Set device and random seed
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    util.set_seed(args.seed)

    # Data generator and model
    datagen = du.PermutedMnistGenerator(
        data_path=args.data_root,
        num_tasks=args.num_tasks,
        download=args.download)
    model = util.get_model(model_class='MLP')
    model.to(device)

    # Initialize Laplace approximation (LA)
    backend = AsdlGGN if args.approx_type == 'ggn' else AsdlEF
    prior_mean = torch.zeros_like(parameters_to_vector(model.parameters()))
    la = Laplace(
        model, 'classification',
        subset_of_weights='all',
        hessian_structure=args.hessian_structure,
        prior_mean=prior_mean,
        prior_precision=args.prior_prec_init,
        backend=backend)

    # Loop through all tasks
    test_loaders = list()
    results = list()
    for task_id in range(args.num_tasks):
        print()
        print(f'Task {task_id+1}')
        print()

        # Get data of new task
        train_loader, test_loader = datagen.next_task(args.batch_size)
        test_loaders.append(test_loader)

        # Train on new task
        train(args, model, la, train_loader, task_id, device)

        # Fit LA to current task
        la.fit(train_loader, override=False)

        # Evaluate on all tasks up to the current task
        test_accs = test(args, la, test_loaders, device)
        results.append(test_accs)
        print()
        print(f'Test accuracies after task {task_id+1}:')
        print(test_accs, np.nanmean(test_accs))
        print()
        print('---------------------------------------------------------------')

    # Save results
    results = np.stack(results)
    if args.run_name is None:
        results_path = f'{args.benchmark}_marglik_{args.hessian_structure}_{args.seed}.npy'
    else:
        results_path = f'{args.run_name}.npy'
    np.save(os.path.join(args.results_root, results_path), results)


def train(args, model, la, train_loader, task_id, device):
    model.train()

    N = len(train_loader.dataset)
    # Set loss function and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    n_steps = args.num_epochs * len(train_loader)
    scheduler = CosineAnnealingLR(optimizer, n_steps, eta_min=args.lr * 1e-3)

    # Train for multiple epochs on current task
    for epoch in range(args.num_epochs):
        train_loss = 0.
        for X, y in train_loader:
            f = model(X.to(device))

            # Subtract log prior from loss function
            mean = parameters_to_vector(model.parameters())
            loss = loss_fn(f, y.to(device)) - args.lam * la.log_prob(mean) / N

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item() * len(X)

        train_loss /= N

        if (epoch + 1) % 5 != 0:
            continue

        marglik = optimize_marglik(la, train_loader)

        print(f'Task {task_id+1} epoch {epoch+1} - train loss: {train_loss:.3f}, neg. log marglik: {marglik:.3f}')


def optimize_marglik(la, train_loader):
    prior_prec = la.prior_precision
    hyper_la = deepcopy(la)
    hyper_la.prior_mean = la.mean
    # Fit LA for marginal likelihood optimization
    hyper_la.fit(train_loader, override=False)
    hyper_la.optimize_prior_precision(init_prior_prec=prior_prec)
    # Include optimized initial prior precision in prior (for regularization)
    la.prior_precision = hyper_la.prior_precision.clone()
    return - hyper_la.log_marginal_likelihood().detach().item()


@torch.no_grad()
def test(args, laplace, test_loaders, device):
    # loop through all tasks up to current task
    test_accs = list()
    for test_loader in test_loaders:
        # get accuracy on task
        correct = 0
        for X, y in test_loader:
            f = laplace(X.to(device))
            correct += (y.to(device) == f.argmax(1)).sum()
        acc = correct.item() / len(test_loader.dataset)
        test_accs.append(acc)

    test_accs.extend([np.nan for _ in range(args.num_tasks - len(test_accs))])

    return np.array(test_accs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', type=str, choices=['permutedMNIST'],
                        default='permutedMNIST', help='name of continual learning benchmark')
    parser.add_argument('--data_root', type=str, default='./data', help='root of dataset')
    parser.add_argument('--results_root', type=str, default='./results', help='root of dataset')
    parser.add_argument('--download', action='store_true',
                        help='if True, downloads the datasets needed for given benchmark')

    parser.add_argument('--num_tasks', type=int, default=10, help='number of tasks')
    parser.add_argument('--lam', type=float, default=1., help='regularization hyper-parameter lambda')
    parser.add_argument('--batch_size', type=int, default=256, help='mini-batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--lr_hyp', type=float, default=1e-1, 
                        help='learning rate for marglik optimization')
    parser.add_argument('--hessian_structure', type=str, choices=['diag', 'kron'], default='kron')
    parser.add_argument('--approx_type', type=str, choices=['ggn', 'ef'], default='ggn')
    parser.add_argument('--prior_structure', default='scalar', choices=['all', 'layerwise', 'scalar'])
    parser.add_argument('--prior_prec_init', default=1e-3, type=float)
    parser.add_argument('--num_epochs', type=int, default=100, help='number of training epochs')
    parser.add_argument('--seed', type=int, default=711, help='random seed')

    parser.add_argument('--config', default=None, nargs='+',
                        help='YAML config file path')
    parser.add_argument('--run_name', type=str, help='overwrite save file name')

    args = parser.parse_args()
    args_dict = vars(args)

    # load config file (YAML)
    if args.config is not None:
        for path in args.config:
            with open(path) as f:
                config = yaml.full_load(f)
            args_dict.update(config)

    for key, val in args_dict.items():
        print(f'{key}: {val}')
    print()

    main(args)
