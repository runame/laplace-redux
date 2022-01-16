import os
import time
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, mean_squared_error
import scipy.stats as st

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.models as torch_models
from torch.nn.utils import parameters_to_vector

import utils.wilds_utils as wu
from baselines.vanilla.models import mlp, lenet, wrn, wrn_fixup
try:
    from baselines.bbb.models import lenet as lenet_bbb
    from baselines.bbb.models import wrn as wrn_bbb
except ModuleNotFoundError:
    pass


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False


def load_pretrained_model(args, model_idx, device):
    """ Choose appropriate architecture and load pre-trained weights """

    if 'WILDS' in args.benchmark:
        dataset = args.benchmark[6:]
        model = wu.load_pretrained_wilds_model(dataset, args.models_root,
                                               device, model_idx, args.model_seed)
    else:
        model = get_model(args.model, no_dropout=args.no_dropout).to(device)
        if args.benchmark in ['R-MNIST', 'MNIST-OOD']:
            fpath = os.path.join(args.models_root, 'lenet_mnist/lenet_mnist_{}_{}')
        elif args.benchmark in ['R-FMNIST', 'FMNIST-OOD']:
            fpath = os.path.join(args.models_root, 'lenet_fmnist/lenet_fmnist_{}.pt')
        elif args.benchmark in ['CIFAR-10-C', 'CIFAR-10-OOD']:
            fpath = os.path.join(args.models_root, 'wrn16_4_cifar10/wrn_16-4_cifar10_{}_{}')
        elif args.benchmark == 'ImageNet-C':
            fpath = os.path.join(args.models_root, 'wrn50_2_imagenet/wrn_50-2_imagenet_{}_{}')

        if args.method == 'csghmc':
            fname = fpath.format(args.model_seed, model_idx+1)
            state_dicts = torch.load(fname, map_location=device)

            for m, state_dict in zip(model, state_dicts):
                m.load_state_dict(state_dict)
                m.to(device)

            if args.data_parallel:
                model = [torch.nn.DataParallel(m) for m in model]
        else:
            if args.model_path is not None:
                model.load_state_dict(torch.load(args.model_path, map_location=device),
                                      strict=False)
            else:
                fname = (fpath.format(args.model_seed, model_idx+1) if 'FMNIST' not in args.benchmark 
                         else fpath.format(args.model_seed))
                load_model = (
                    model.net
                    if args.benchmark in ['R-MNIST', 'MNIST-OOD'] and 'BBB' not in args.model
                    else model
                )
                load_model.load_state_dict(torch.load(fname , map_location=device))
            model.to(device)

    if args.data_parallel and (args.method != 'csghmc'):
        model = torch.nn.DataParallel(model)
    return model


def get_model(model_class, no_dropout=False):
    if model_class == 'MLP':
        model = mlp.MLP([784, 100, 100, 10], act='relu')
    elif model_class == 'LeNet':
        model = lenet.LeNet()
    elif model_class == 'LeNet-BBB-reparam':
        model = lenet_bbb.LeNetBBB(estimator='reparam')
    elif model_class == 'LeNet-BBB-flipout':
        model = lenet_bbb.LeNetBBB(estimator='flipout')
    elif model_class == 'LeNet-CSGHMC':
        model = [lenet.LeNet() for _ in range(12)]  # 12 samples in CSGHMC
    elif model_class == 'WRN16-4':
        model = wrn.WideResNet(16, 4, 10, dropRate=0.3)
    elif model_class == 'WRN16-4-fixup':
        model = wrn_fixup.FixupWideResNet(16, 4, 10, dropRate=0.0 if no_dropout else 0.2)
    elif model_class == 'WRN16-4-BBB-reparam':
        model = wrn_bbb.WideResNetBBB(16, 4, 10, estimator='reparam')
    elif model_class == 'WRN16-4-BBB-flipout':
        model = wrn_bbb.WideResNetBBB(16, 4, 10, estimator='flipout')
    elif model_class == 'WRN16-4-CSGHMC':
        model = [wrn.WideResNet(16, 4, 10, dropRate=0) for _ in range(12)]  # 12 samples in CSGHMC
    elif model_class == 'WRN50-2':
        model = torch_models.wide_resnet50_2()
    else:
        raise ValueError('Choose LeNet, WRN16-4, or WRN50-2 as model_class.')
    return model


def mixture_model_pred(components, x, mixture_weights, prediction_mode='mola',
                       pred_type='glm', link_approx='probit', n_samples=100,
                       likelihood='classification'):
    if prediction_mode == 'ensemble':
        return ensemble_pred(components, x, likelihood=likelihood)

    out = 0.  # out will be a tensor
    for model, pi in zip(components, mixture_weights):
        if prediction_mode == 'mola':
            out_prob = model(x, pred_type=pred_type, n_samples=n_samples, link_approx=link_approx)
        elif prediction_mode == 'multi-swag':
            from baselines.swag.swag import predict_swag
            swag_model, swag_samples, swag_bn_params = model
            out_prob = predict_swag(swag_model, x, swag_samples, swag_bn_params)
        else:
            raise ValueError('For now only ensemble, mola, and multi-swag are supported.')
        out += pi * out_prob
    return out


def ensemble_pred(components, x, likelihood='classification'):
    """ Make predictions for deep ensemble """

    outs = []
    for model in components:
        model.eval()
        out_prob = model(x).detach()
        if likelihood == 'classification':
            out_prob = out_prob.softmax(1)
        outs.append(out_prob)

    outs = torch.stack(outs, dim=0)
    out_mean = torch.mean(outs, dim=0)

    if likelihood == 'regression':
        out_var = torch.var(outs, dim=0).unsqueeze(2)
        return [out_mean, out_var]
    else:
        return out_mean


def expand_prior_precision(prior_prec, model):
    theta = parameters_to_vector(model.parameters())
    device, P = theta.device, len(theta)
    assert prior_prec.ndim == 1
    if len(prior_prec) == 1:  # scalar
        return torch.ones(P, device=device) * prior_prec
    elif len(prior_prec) == P:  # full diagonal
        return prior_prec.to(device)
    else:
        return torch.cat([delta * torch.ones_like(m).flatten() for delta, m
                          in zip(prior_prec, model.parameters())])


def prior_prec_to_tensor(args, prior_prec, model):
    H = len(list(model.parameters()))
    theta = parameters_to_vector(model.parameters())
    device, P = theta.device, len(theta)
    if args.prior_structure == 'scalar':
        log_prior_prec = torch.ones(1, device=device)
    elif args.prior_structure == 'layerwise':
        log_prior_prec = torch.ones(H, device=device)
    elif args.prior_structure == 'all':
        log_prior_prec = torch.ones(P, device=device)
    else:
        raise ValueError(f'Invalid prior structure {args.prior_structure}')
    return log_prior_prec * prior_prec


def get_auroc(py_in, py_out):
    py_in, py_out = py_in.cpu().numpy(), py_out.cpu().numpy()
    labels = np.zeros(len(py_in)+len(py_out), dtype='int32')
    labels[:len(py_in)] = 1
    examples = np.concatenate([py_in.max(1), py_out.max(1)])
    return roc_auc_score(labels, examples)


def get_fpr95(py_in, py_out):
    py_in, py_out = py_in.cpu().numpy(), py_out.cpu().numpy()
    conf_in, conf_out = py_in.max(1), py_out.max(1)
    tpr = 95
    perc = np.percentile(conf_in, 100-tpr)
    fp = np.sum(conf_out >=  perc)
    fpr = np.sum(conf_out >=  perc) / len(conf_out)
    return fpr.item(), perc.item()


def get_brier_score(probs, targets):
    targets = F.one_hot(targets, num_classes=probs.shape[1])
    return torch.mean(torch.sum((probs - targets)**2, axis=1)).item()


def get_calib(pys, y_true, M=100):
    pys, y_true = pys.cpu().numpy(), y_true.cpu().numpy()
    # Put the confidence into M bins
    _, bins = np.histogram(pys, M, range=(0, 1))

    labels = pys.argmax(1)
    confs = np.max(pys, axis=1)
    conf_idxs = np.digitize(confs, bins)

    # Accuracy and avg. confidence per bin
    accs_bin = []
    confs_bin = []
    nitems_bin = []

    for i in range(M):
        labels_i = labels[conf_idxs == i]
        y_true_i = y_true[conf_idxs == i]
        confs_i = confs[conf_idxs == i]

        acc = np.nan_to_num(np.mean(labels_i == y_true_i), 0)
        conf = np.nan_to_num(np.mean(confs_i), 0)

        accs_bin.append(acc)
        confs_bin.append(conf)
        nitems_bin.append(len(labels_i))

    accs_bin, confs_bin = np.array(accs_bin), np.array(confs_bin)
    nitems_bin = np.array(nitems_bin)

    ECE = np.average(np.abs(confs_bin-accs_bin), weights=nitems_bin/nitems_bin.sum())
    MCE = np.max(np.abs(accs_bin - confs_bin))

    return ECE, MCE


def get_calib_regression(pred_means, pred_stds, y_true, return_hist=False, M=10):
    '''
    Kuleshov et al. ICML 2018, eq. 9
    * pred_means, pred_stds, y_true must be np.array's
    * Set return_hist to True to also return the "histogram"---useful for visualization (see paper)
    '''
    T = len(y_true)
    ps = np.linspace(0, 1, M)
    cdf_vals = [st.norm(m, s).cdf(y_t) for m, s, y_t in zip(pred_means, pred_stds, y_true)]
    p_hats = np.array([len(np.where(cdf_vals <= p)[0]) / T for p in ps])
    cal = T*mean_squared_error(ps, p_hats)  # Sum-squared-error

    return (cal, ps, p_hats) if return_hist else cal


def get_sharpness(pred_stds):
    '''
    Kuleshov et al. ICML 2018, eq. 10

    pred_means be np.array
    '''
    return np.mean(pred_stds**2)


def timing(fun):
    """
    Return the original output(s) and a wall-clock timing in second.
    """
    if torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        ret = fun()
        end.record()
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end)/1000
    else:
        start_time = time.time()
        ret = fun()
        end_time = time.time()
        elapsed_time = end_time - start_time
    return ret, elapsed_time


def save_results(args, metrics):
    """ Save the computed metrics """

    if args.run_name is None:
        res_str = f'_{args.subset_of_weights}_{args.hessian_structure}' if args.method in ['laplace', 'mola'] else ''
        temp_str = '' if args.temperature == 1.0 else f'_{args.temperature}'
        method_str = f'temp' if args.use_temperature_scaling and args.method == 'map' else args.method
        frac_str = f'_{args.data_fraction}' if args.data_fraction < 1.0 else ''
        result_path = f'./results/{args.benchmark}/{method_str}{res_str}{temp_str}_{args.model_seed}{frac_str}.npy'
    else:
        result_path = f'./results/{args.run_name}.npy'
    Path(result_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving results to {result_path}...")
    np.save(result_path, metrics)


def get_prior_precision(args, device):
    """ Obtain the prior precision parameter from the cmd arguments """

    if type(args.prior_precision) is str: # file path
        prior_precision = torch.load(args.prior_precision, map_location=device)
    elif type(args.prior_precision) is float:
        prior_precision = args.prior_precision
    else:
        raise ValueError('Algorithm not happy with inputs prior precision :(')

    return prior_precision
