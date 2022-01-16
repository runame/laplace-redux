import torch
import yaml

from laplace import KronLaplace, DiagLaplace, DiagLLLaplace, KronLLLaplace, FullLLLaplace, FullLaplace
from laplace.curvature import AsdlGGN, AsdlEF, BackPackGGN, BackPackEF

from marglik_training.marglik_optimization import marglik_optimization
from utils.data_utils import get_mnist_loaders, get_cifar10_loaders, get_fmnist_loaders
from utils.utils import set_seed, get_model


def get_laplace_class(flavor, last_layer):
    if flavor == 'diag':
        if last_layer:
            return DiagLLLaplace
        else:
            return DiagLaplace
    elif flavor == 'kron':
        if last_layer:
            return KronLLLaplace
        else:
            return KronLaplace
    elif flavor == 'full':
        if last_layer:
            return FullLLLaplace
        else:
            return FullLaplace
    else:
        raise ValueError()


def get_backend(backend, approx_type):
    if backend == 'kazuki':
        if approx_type == 'ggn':
            return AsdlGGN
        else:
            return AsdlEF
    elif backend == 'backpack':
        if approx_type == 'ggn':
            return BackPackGGN
        else:
            return BackPackEF
    else:
        raise ValueError()


if __name__ == "__main__":
    import argparse
    import logging
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['MNIST', 'FMNIST', 'CIFAR-10-NODA', 'CIFAR-10'], default='MNIST')
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_min', type=float, default=1e-5, help='minimum lr decayed to.')
    parser.add_argument('--lr_hyp', type=float, default=1e-1)
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam'], default='Adam')
    parser.add_argument('--prior_structure', type=str, choices=['layerwise', 'diagonal', 'scalar'],
                        help='structure of the Gaussian prior.', default='layerwise')
    parser.add_argument('--temperature', type=float, default=1.,
                        help='temperature scaling the likelihood, higher leads to more concentration.')
    parser.add_argument('--laplace_flavor', type=str, choices=['diag', 'kron', 'full'], default='kron')
    parser.add_argument('--approx_type', type=str, choices=['ggn', 'ef'], default='ggn')
    parser.add_argument('--backend', type=str, choices=['backpack', 'kazuki'], default='kazuki')
    parser.add_argument('--decay', type=str, choices=['exp', 'cos'], default='exp')
    parser.add_argument('--last_layer', action='store_true')
    parser.add_argument('--no_dropout', action='store_true', help='only matters for WRN.')
    parser.add_argument('--F', type=int, default=1, help='marginal likelihood frequency')
    parser.add_argument('--K', type=int, default=100, help='marginal likelihood steps')
    parser.add_argument('--B', type=int, default=0, help='marginal likelihood burnin')
    parser.add_argument('--data_root', type=str, default='./data', help='root of dataset')
    parser.add_argument('--seed', default=711, type=int)
    parser.add_argument('--nolog', action='store_true')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--config', default=None, nargs='+', help='YAML config file path.')
    parser.add_argument('--run_name', default=None, help='Fix the save file name.')
    args = parser.parse_args()
    args_dict = vars(args)
    logging.basicConfig(level=logging.WARNING if args.nolog else logging.INFO)

    if args.config is not None:
        for config_file in args.config:
            with open(config_file) as f:
                config = yaml.full_load(f)
            args_dict.update(config)
    print(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)
    Laplace = get_laplace_class(args.laplace_flavor, args.last_layer)
    Backend = get_backend(args.backend, args.approx_type)
    if args.dataset == 'MNIST':
        train_loader, valid_loader, _ = get_mnist_loaders(args.data_root, train_batch_size=args.batch_size, download=args.download, device=device)
        set_seed(args.seed)
        model = get_model('LeNet').to(device)
        lap, model, _, _ = marglik_optimization(model, train_loader, prior_structure=args.prior_structure, scheduler=args.decay,
                                                n_epochs=args.n_epochs, lr=args.lr, lr_min=args.lr_min, n_epochs_burnin=args.B,
                                                n_hypersteps=args.K, marglik_frequency=args.F, lr_hyp=args.lr_hyp,
                                                laplace=Laplace, backend=Backend, temperature=args.temperature, optimizer=args.optimizer,
                                                valid_loader=valid_loader, prior_prec_init=30.)
        # 30 prior prec init == 5e-4 * N=60000 for MNIST/FMNIST.
    elif args.dataset == 'FMNIST':
        train_loader, valid_loader, _ = get_fmnist_loaders(args.data_root, train_batch_size=args.batch_size, download=args.download, device=device)
        set_seed(args.seed)
        model = get_model('LeNet').to(device)
        lap, model, _, _ = marglik_optimization(model, train_loader, prior_structure=args.prior_structure, scheduler=args.decay,
                                                n_epochs=args.n_epochs, lr=args.lr, lr_min=args.lr_min, n_epochs_burnin=args.B,
                                                n_hypersteps=args.K, marglik_frequency=args.F, lr_hyp=args.lr_hyp,
                                                laplace=Laplace, backend=Backend, temperature=args.temperature, optimizer=args.optimizer,
                                                valid_loader=valid_loader, prior_prec_init=30.)

    elif args.dataset == 'CIFAR-10-NODA':
        train_loader, valid_loader, _ = get_cifar10_loaders(args.data_root, train_batch_size=args.batch_size, download=args.download, data_augmentation=False)
        set_seed(args.seed)
        model = get_model('WRN16-4-fixup', no_dropout=args.no_dropout).to(device)
        lap, model, _, _ = marglik_optimization(model, train_loader, prior_structure=args.prior_structure, scheduler=args.decay,
                                                n_epochs=args.n_epochs, lr=args.lr, lr_min=args.lr_min, n_epochs_burnin=args.B,
                                                n_hypersteps=args.K, marglik_frequency=args.F, lr_hyp=args.lr_hyp,
                                                laplace=Laplace, backend=Backend, temperature=args.temperature, optimizer=args.optimizer,
                                                valid_loader=valid_loader, prior_prec_init=25.)
    elif args.dataset == 'CIFAR-10':
        train_loader, valid_loader, _ = get_cifar10_loaders(args.data_root, train_batch_size=args.batch_size, download=args.download, data_augmentation=True)
        set_seed(args.seed)
        model = get_model('WRN16-4-fixup', no_dropout=args.no_dropout).to(device)
        lap, model, _, _ = marglik_optimization(model, train_loader, prior_structure=args.prior_structure, scheduler=args.decay,
                                                n_epochs=args.n_epochs, lr=args.lr, lr_min=args.lr_min, n_epochs_burnin=args.B,
                                                n_hypersteps=args.K, marglik_frequency=args.F, lr_hyp=args.lr_hyp,
                                                laplace=Laplace, backend=Backend, temperature=args.temperature, optimizer=args.optimizer,
                                                valid_loader=valid_loader, prior_prec_init=25.)
        # 25 prior prec init == 5e-4 * N=50000 for CIFAR-10.

    if args.run_name is None:
        file_str = f'models/{args.dataset}_{args.laplace_flavor}_{args.prior_structure}_{args.approx_type}_{args.seed}'
        file_str += 'll' if args.last_layer else ''
    else:
        file_str = 'models/' + args.run_name
    torch.save(model.to('cpu').state_dict(), file_str + '_model.pt')
    torch.save(lap.prior_precision.to('cpu'), file_str + '_delta.pt')
