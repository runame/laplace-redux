from tqdm import tqdm
import torch
from torch import nn

from utils.utils import mixture_model_pred


@torch.no_grad()
def test(components, test_loader, prediction_mode, pred_type='glm', n_samples=100,
         link_approx='probit', no_loss_acc=False, device='cpu',
         likelihood='classification', sigma_noise=None):

    temperature_scaling_model = None
    if prediction_mode in ['map', 'laplace', 'bbb', 'csghmc']:
        model = components[0]
        if prediction_mode in ['map', 'bbb']:
            if prediction_mode == 'map' and isinstance(model, tuple):
                model, temperature_scaling_model = model[0], model[1]
            model.eval()
        elif prediction_mode == 'csghmc':
            for m in model:
                m.eval()
    elif prediction_mode == 'swag':
        model, swag_samples, swag_bn_params = components[0]

    if likelihood == 'regression' and sigma_noise is None:
        raise ValueError('Must provide sigma_noise for regression!')

    if likelihood == 'classification':
        loss_fn = nn.NLLLoss()
    elif likelihood == 'regression':
        loss_fn = nn.GaussianNLLLoss(full=True)
    else:
        raise ValueError(f'Invalid likelihood type {likelihood}')

    all_y_true = list()
    all_y_prob = list()
    all_y_var = list()
    for data in tqdm(test_loader):
        x, y = data[0].to(device), data[1].to(device)
        all_y_true.append(y.cpu())

        if prediction_mode in ['ensemble', 'mola', 'multi-swag']:
            # set uniform mixture weights
            K = len(components)
            pi = torch.ones(K, device=device) / K
            y_prob = mixture_model_pred(
                components, x, pi,
                prediction_mode=prediction_mode,
                pred_type=pred_type,
                link_approx=link_approx,
                n_samples=n_samples,
                likelihood=likelihood)

        elif prediction_mode == 'laplace':
            y_prob = model(
                x, pred_type=pred_type, link_approx=link_approx, n_samples=n_samples)

        elif prediction_mode == 'map':
            # y_prob here is logits since we need them for temp. scaling
            y_prob = model(x).detach()

        elif prediction_mode == 'bbb':
            y_prob = torch.stack([model(x)[0].softmax(-1) for _ in range(10)]).mean(0)

        elif prediction_mode == 'csghmc':
            y_prob = torch.stack([m(x).softmax(-1) for m in model]).mean(0)

        elif prediction_mode == 'swag':
            from baselines.swag.swag import predict_swag
            y_prob = predict_swag(model, x, swag_samples, swag_bn_params)

        else:
            raise ValueError(
                'Choose one out of: map, ensemble, laplace, mola, bbb, csghmc, swag, multi-swag.')

        if likelihood == 'regression':
            y_mean = y_prob if prediction_mode == 'map' else y_prob[0]
            y_var = torch.zeros_like(y_mean) if prediction_mode == 'map' else y_prob[1].squeeze(2)
            all_y_prob.append(y_mean.cpu())
            all_y_var.append(y_var.cpu())
        else:
            all_y_prob.append(y_prob.cpu())

    # aggregate predictive distributions, true labels and metadata
    all_y_prob = torch.cat(all_y_prob, dim=0)
    all_y_true = torch.cat(all_y_true, dim=0)

    if temperature_scaling_model is not None:
        print('Calibrating predictions using temperature scaling...')
        all_y_prob = torch.from_numpy(temperature_scaling_model.predict_proba(all_y_prob.numpy()))

    elif prediction_mode == 'map' and likelihood == 'classification':
        all_y_prob = all_y_prob.softmax(dim=1)

    # compute some metrics: mean confidence, accuracy and negative log-likelihood
    metrics = {}
    if likelihood == 'classification':
        assert all_y_prob.sum(-1).mean() == 1, '`all_y_prob` are logits but probs. are required'
        c, preds = torch.max(all_y_prob, 1)
        metrics['conf'] = c.mean().item()

    if not no_loss_acc:
        if likelihood == 'regression':
            all_y_var = torch.cat(all_y_var, dim=0) + sigma_noise**2
            metrics['nll'] = loss_fn(all_y_prob, all_y_true, all_y_var).item()

        else:
            all_y_var = None
            metrics['nll'] = loss_fn(all_y_prob.log(), all_y_true).item()
            metrics['acc'] = (all_y_true == preds).float().mean().item()

    return metrics, all_y_prob, all_y_var


@torch.no_grad()
def predict(dataloader, model):
    py = []

    for x, y in dataloader:
        x = x.cuda()
        py.append(torch.softmax(model(x), -1))

    return torch.cat(py, dim=0)


@torch.no_grad()
def predict_ensemble(dataloader, models):
    py = []

    for x, y in dataloader:
        x = x.cuda()

        _py = 0
        for model in models:
            _py += 1/len(models) * torch.softmax(model(x), -1)
        py.append(_py)

    return torch.cat(py, dim=0)


@torch.no_grad()
def predict_vb(dataloader, model, n_samples=1):
    py = []

    for x, y in dataloader:
        x = x.cuda()

        _py = 0
        for _ in range(n_samples):
            f_s, _ = model(x)  # The second return is KL
            _py += torch.softmax(f_s, 1)
        _py /= n_samples

        py.append(_py)

    return torch.cat(py, dim=0)
