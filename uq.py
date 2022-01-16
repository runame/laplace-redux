import argparse
import yaml

import torch
import pycalib
from laplace import Laplace

import utils.data_utils as du
import utils.wilds_utils as wu
import utils.utils as util
from utils.test import test
from marglik_training.train_marglik import get_backend
from baselines.swag.swag import fit_swag_and_precompute_bn_params

import warnings
warnings.filterwarnings('ignore')


def main(args):
    # set device and random seed
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args.prior_precision = util.get_prior_precision(args, device)
    util.set_seed(args.seed)

    # load in-distribution data
    in_data_loaders, ids, no_loss_acc = du.get_in_distribution_data_loaders(
        args, device)
    train_loader, val_loader, in_test_loader = in_data_loaders

    # fit models
    mixture_components = fit_models(args, train_loader, val_loader, device)

    # evaluate models
    metrics = evaluate_models(
        args, mixture_components, in_test_loader, ids, no_loss_acc, device)

    # save results
    util.save_results(args, metrics)


def fit_models(args, train_loader, val_loader, device):
    """ load pre-trained weights, fit inference methods, and tune hyperparameters """

    mixture_components = list()
    for model_idx in range(args.nr_components):
        model = util.load_pretrained_model(args, model_idx, device)

        if args.method in ['laplace', 'mola']:
            if type(args.prior_precision) is str: # file path
                prior_precision = torch.load(args.prior_precision, map_location=device)
            elif type(args.prior_precision) is float:
                prior_precision = args.prior_precision
            else:
                raise ValueError('prior precision has to be either float or string (file path)')
            Backend = get_backend(args.backend, args.approx_type)
            optional_args = dict()

            if args.subset_of_weights == 'last_layer':
                optional_args['last_layer_name'] = args.last_layer_name

            print('Fitting Laplace approximation...')
            
            model = Laplace(model, args.likelihood,
                            subset_of_weights=args.subset_of_weights,
                            hessian_structure=args.hessian_structure,
                            prior_precision=prior_precision,
                            temperature=args.temperature,
                            backend=Backend, **optional_args)
            model.fit(train_loader)

            if (args.optimize_prior_precision is not None) and (args.method == 'laplace'):
                if (type(prior_precision) is float) and (args.prior_structure != 'scalar'):
                    n = model.n_params if args.prior_structure == 'all' else model.n_layers
                    prior_precision = prior_precision * torch.ones(n, device=device)
                
                print('Optimizing prior precision for Laplace approximation...')

                verbose_prior = args.prior_structure == 'scalar'
                model.optimize_prior_precision(
                    method=args.optimize_prior_precision,
                    init_prior_prec=prior_precision,
                    val_loader=val_loader,
                    pred_type=args.pred_type,
                    link_approx=args.link_approx,
                    n_samples=args.n_samples,
                    verbose=verbose_prior
                )

        elif args.method in ['swag', 'multi-swag']:
            print("Fitting SWAG...")

            model = fit_swag_and_precompute_bn_params(
                model, device, train_loader, args.swag_n_snapshots,
                args.swag_lr, args.swag_c_epochs, args.swag_c_batches, 
                args.data_parallel, args.n_samples, args.swag_bn_update_subset)

        elif (args.method == 'map' and args.likelihood == 'classification' 
              and args.use_temperature_scaling):
            print("Fitting temperature scaling model on validation data...")
            all_y_prob = [model(d[0].to(device)).detach().cpu() for d in val_loader]
            all_y_prob = torch.cat(all_y_prob, dim=0)
            all_y_true = torch.cat([d[1] for d in val_loader], dim=0)

            temperature_scaling_model = pycalib.calibration_methods.TemperatureScaling()
            temperature_scaling_model.fit(all_y_prob.numpy(), all_y_true.numpy())
            model = (model, temperature_scaling_model)

        if args.likelihood == 'regression' and args.sigma_noise is None:
            print("Optimizing noise standard deviation on validation data...")
            args.sigma_noise = wu.optimize_noise_standard_deviation(model, val_loader, device)

        mixture_components.append(model)

    return mixture_components


def evaluate_models(args, mixture_components, in_test_loader, ids, no_loss_acc, device):
    """ evaluate the models and return relevant evaluation metrics """

    metrics = []
    for i, id in enumerate(ids):
        # load test data
        test_loader = in_test_loader if i == 0 else du.get_ood_test_loader(
            args, id)

        # make model predictions and compute some metrics
        test_output, test_time = util.timing(lambda: test(
            mixture_components, test_loader, args.method,
            pred_type=args.pred_type, link_approx=args.link_approx,
            n_samples=args.n_samples, device=device, no_loss_acc=no_loss_acc,
            likelihood=args.likelihood, sigma_noise=args.sigma_noise))
        some_metrics, all_y_prob, all_y_var = test_output
        some_metrics['test_time'] = test_time

        if i == 0:
            all_y_prob_in = all_y_prob.clone()

        # compute more metrics, aggregate and print them:
        # log likelihood, accuracy, confidence, Brier sore, ECE, MCE, AUROC, FPR95
        more_metrics = compute_metrics(
            i, id, all_y_prob, test_loader, all_y_prob_in, all_y_var, args)
        metrics.append({**some_metrics, **more_metrics})
        print(', '.join([f'{k}: {v:.4f}' for k, v in metrics[-1].items()]))

    return metrics


def compute_metrics(i, id, all_y_prob, test_loader, all_y_prob_in, all_y_var, args):
    """ compute evaluation metrics """

    metrics = {}

    # compute Brier, ECE and MCE for distribution shift and WILDS benchmarks
    if args.benchmark in ['R-MNIST', 'R-FMNIST', 'CIFAR-10-C', 'ImageNet-C'] and (args.benchmark != 'WILDS-poverty'):
        print(f'{args.benchmark} with distribution shift intensity {i}')
        labels = torch.cat([data[1] for data in test_loader])
        metrics['brier'] = util.get_brier_score(all_y_prob, labels)
        metrics['ece'], metrics['mce'] = util.get_calib(all_y_prob, labels)

    # compute AUROC and FPR95 for OOD benchmarks
    if args.benchmark in ['MNIST-OOD', 'FMNIST-OOD', 'CIFAR-10-OOD']:
        print(f'{args.benchmark} - dataset: {id}')
        if i > 0:
            # compute other metrics
            metrics['auroc'] = util.get_auroc(all_y_prob_in, all_y_prob)
            metrics['fpr95'], _ = util.get_fpr95(all_y_prob_in, all_y_prob)

    # compute regression calibration
    if args.benchmark == "WILDS-poverty":
        print(f'{args.benchmark} with distribution shift intensity {i}')
        labels = torch.cat([data[1] for data in test_loader])
        metrics['calib_regression'] = util.get_calib_regression(
            all_y_prob.numpy(), all_y_var.sqrt().numpy(), labels.numpy())

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', type=str,
                        choices=['R-MNIST', 'R-FMNIST', 'CIFAR-10-C', 'ImageNet-C',
                                 'MNIST-OOD', 'FMNIST-OOD', 'CIFAR-10-OOD',
                                 'WILDS-camelyon17', 'WILDS-iwildcam',
                                 'WILDS-civilcomments', 'WILDS-amazon',
                                 'WILDS-fmow', 'WILDS-poverty'],
                        default='CIFAR-10-C', help='name of benchmark')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='root of dataset')
    parser.add_argument('--download', action='store_true',
                        help='if True, downloads the datasets needed for given benchmark')
    parser.add_argument('--data_fraction', type=float, default=1.0,
                    help='fraction of data to use (only supported for WILDS)')
    parser.add_argument('--models_root', type=str, default='./models',
                        help='root of pre-trained models')
    parser.add_argument('--model_seed', type=int, default=None,
                        help='random seed with which model(s) were trained')
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--hessians_root', type=str, default='./hessians',
                        help='root of pre-computed Hessians')
    parser.add_argument('--method', type=str,
                        choices=['map', 'ensemble',
                                 'laplace', 'mola',
                                 'swag', 'multi-swag',
                                 'bbb', 'csghmc'],
                        default='laplace',
                        help='name of method to use')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed')

    parser.add_argument('--pred_type', type=str,
                        choices=['nn', 'glm'],
                        default='glm',
                        help='type of approximation of predictive distribution')
    parser.add_argument('--link_approx', type=str,
                        choices=['mc', 'probit', 'bridge'],
                        default='probit',
                        help='type of approximation of link function')
    parser.add_argument('--n_samples', type=int, default=100,
                        help='nr. of MC samples for approximating the predictive distribution')

    parser.add_argument('--likelihood', type=str, choices=['classification', 'regression'],
                        default='classification', help='likelihood for Laplace')
    parser.add_argument('--subset_of_weights', type=str, choices=['last_layer', 'all'],
                        default='last_layer', help='subset of weights for Laplace')
    parser.add_argument('--backend', type=str, choices=['backpack', 'kazuki'], default='backpack')
    parser.add_argument('--approx_type', type=str, choices=['ggn', 'ef'], default='ggn')
    parser.add_argument('--hessian_structure', type=str, choices=['diag', 'kron', 'full'],
                        default='kron', help='structure of the Hessian approximation')
    parser.add_argument('--last_layer_name', type=str, default=None,
                        help='name of the last layer of the model')
    parser.add_argument('--prior_precision', default=1.,
                        help='prior precision to use for computing the covariance matrix')
    parser.add_argument('--optimize_prior_precision', default=None,
                        choices=['marglik', 'nll'],
                        help='optimize prior precision according to specified method')
    parser.add_argument('--prior_structure', type=str, default='scalar',
                        choices=['scalar', 'layerwise', 'all'])
    parser.add_argument('--sigma_noise', type=float, default=None,
                        help='noise standard deviation for regression (if -1, optimize it)')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='temperature of the likelihood.')

    parser.add_argument('--swag_n_snapshots', type=int, default=40,
                        help='number of snapshots for [Multi]SWAG')
    parser.add_argument('--swag_c_batches', type=int, default=None,
                        help='number of batches between snapshots for [Multi]SWAG')
    parser.add_argument('--swag_c_epochs', type=int, default=1,
                        help='number of epochs between snapshots for [Multi]SWAG')
    parser.add_argument('--swag_lr', type=float, default=1e-2,
                        help='learning rate for [Multi]SWAG')
    parser.add_argument('--swag_bn_update_subset', type=float, default=1.0,
                        help='fraction of train data for updating the BatchNorm statistics for [Multi]SWAG')

    parser.add_argument('--nr_components', type=int, default=1,
                        help='number of mixture components to use')
    parser.add_argument('--mixture_weights', type=str,
                        choices=['uniform', 'optimize'],
                        default='uniform',
                        help='how the mixture weights for MoLA are chosen')

    parser.add_argument('--model', type=str, default='WRN16-4',
                        choices=['LeNet', 'WRN16-4', 'WRN16-4-fixup', 'WRN50-2',
                                 'LeNet-BBB-reparam', 'LeNet-BBB-flipout', 'LeNet-CSGHMC',
                                 'WRN16-4-BBB-reparam', 'WRN16-4-BBB-flipout', 'WRN16-4-CSGHMC'],
                         help='the neural network model architecture')
    parser.add_argument('--no_dropout', action='store_true', help='only for WRN-fixup.')
    parser.add_argument('--data_parallel', action='store_true',
                        help='if True, use torch.nn.DataParallel(model)')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='batch size for testing')
    parser.add_argument('--val_set_size', type=int, default=2000,
                        help='size of validation set (taken from test set)')
    parser.add_argument('--use_temperature_scaling', default=False,
                        help='if True, calibrate model using temperature scaling')

    parser.add_argument('--job_id', type=int, default=0,
                        help='job ID, leave at 0 when running locally')
    parser.add_argument('--config', default=None, nargs='+',
                        help='YAML config file path')
    parser.add_argument('--run_name', type=str, help='overwrite save file name')
    parser.add_argument('--noda', action='store_true')

    args = parser.parse_args()
    args_dict = vars(args)

    # load config file (YAML)
    if args.config is not None:
        for path in args.config:
            with open(path) as f:
                config = yaml.full_load(f)
            args_dict.update(config)

    if args.data_parallel and (args.method in ['laplace, mola']):
        raise NotImplementedError(
            'laplace and mola do not support DataParallel yet.')

    if (args.optimize_prior_precision is not None) and (args.method == 'mola'):
        raise NotImplementedError(
            'optimizing the prior precision for MoLA is not supported yet.')

    if args.mixture_weights != 'uniform':
        raise NotImplementedError(
            'Only uniform mixture weights are supported for now.')

    if ((args.method in ['ensemble', 'mola', 'multi-swag']) 
        and (args.nr_components <= 1)):
        parser.error(
            'Choose nr_components > 1 for ensemble, MoLA, or MultiSWAG.')

    if args.model != 'WRN16-4-fixup' and args.no_dropout:
        parser.error(
            'No dropout option only available for Fixup.')

    if args.benchmark in ['R-MNIST', 'MNIST-OOD', 'R-FMNIST', 'FMNIST-OOD']:
        if 'LeNet' not in args.model:
            parser.error("Only LeNet works for R-MNIST.")
    elif args.benchmark in ['CIFAR-10-C', 'CIFAR-10-OOD']:
        if 'WRN16-4' not in args.model:
            parser.error("Only WRN16-4 works for CIFAR-10-C.")
    elif args.benchmark == 'ImageNet-C':
        if not (args.model == 'WRN50-2'):
            parser.error("Only WRN50-2 works for ImageNet-C.")

    if args.benchmark == "WILDS-poverty":
        args.likelihood = "regression"
    else:
        args.likelihood = "classification"

    for key, val in args_dict.items():
        print(f'{key}: {val}')
    print()

    main(args)
