import os
import argparse


def main(base_config, config, temperature, seed, noda):
    config_name = config.split('/')[-1].split('.')[0]  # laplace type, sparsity, ggn/ef, prior
    if noda: # no DA
        assert temperature == 1
        run_name = 'CIFARNODA' + '_' + config_name + f'_{seed}'
    else:
        run_name = 'CIFAR' + '_' + config_name + f'_{temperature}_{seed}'
    cmd_train = f'python marglik_training/train_marglik.py --run_name {run_name} --config {base_config} {config} --seed {seed} --temperature {temperature}'
    print('RUN:', cmd_train)
    os.system(cmd_train)
    if 'map' in base_config and not 'marglik' in base_config:
        return
    model_path = 'models/' + run_name + '_model.pt'
    delta_path = 'models/' + run_name + '_delta.pt'
    for benchmark in ['CIFAR-10-OOD', 'CIFAR-10-C']:
        for pred in ['map', 'mc', 'probit', 'bridge']:
            bs = 16
            meth = 'map' if pred == 'map' else 'laplace'
            pred_cli = f'--method {meth}' if meth == 'map' else f'--method {meth} --link_approx {pred}'
            if noda:
                benchmark_name = 'CIFARNODA' + benchmark[5:]
                name = f'{benchmark_name}_{config_name}_{pred}_onmarglik_{seed}'
                flag = '--noda'
            else:
                name = f'{benchmark}_{config_name}_{pred}_onmarglik_{temperature}_{seed}'
                flag = ''
            cmd = f'python uq.py --benchmark {benchmark} --model WRN16-4-fixup {pred_cli} --run_name {name} --prior_precision {delta_path} --model_path {model_path} --config {config} --seed 711 --batch_size {bs} --temperature {temperature} {flag}'
            print('SUBMIT:', cmd)
            os.system(cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type=int, required=True)
    parser.add_argument('--base_config', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--temperature', type=float, required=True)
    parser.add_argument('--noda', action='store_true')
    args = parser.parse_args()
    main(args.base_config, args.config, args.temperature, args.seed, args.noda)
