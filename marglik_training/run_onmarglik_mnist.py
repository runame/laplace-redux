import os
import argparse


def main(base_config, config, seed):
    config_name = config.split('/')[-1].split('.')[0]  # laplace type, sparsity, ggn/ef, prior
    run_name = 'MNIST' + '_' + config_name + f'_{seed}'
    cmd_train = f'python marglik_training/train_marglik.py --run_name {run_name} --config {base_config} {config} --seed {seed}'
    print('RUN:', cmd_train)
    os.system(cmd_train)
    model_path = 'models/' + run_name + '_model.pt'
    delta_path = 'models/' + run_name + '_delta.pt'
    for benchmark in ['R-MNIST', 'MNIST-OOD']:
        for pred in ['map', 'mc', 'probit', 'bridge']:
            meth = 'map' if pred == 'map' else 'laplace'
            pred_cli = f'--method {meth}' if meth == 'map' else f'--method {meth} --link_approx {pred}'
            name = f'{benchmark}_{config_name}_{pred}_onmarglik_{seed}'
            cmd = f'python uq.py --benchmark {benchmark} --model LeNet {pred_cli} --run_name {name} --prior_precision {delta_path} --model_path {model_path} --config {config} --seed 711'
            print('SUBMIT:', cmd)
            os.system(cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type=int, required=True)
    parser.add_argument('--base_config', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    main(args.base_config, args.config, args.seed)
