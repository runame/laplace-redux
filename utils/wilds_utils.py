""" Utility methods for WILDS benchmark experiments """

import torch
from torch import nn

from argparse import Namespace
from pathlib import Path
import urllib.request

try:
    from configs.utils import populate_defaults
    from examples.models.initializer import initialize_model
    from examples.transforms import initialize_transform
    from wilds import get_dataset
    from wilds.common.data_loaders import get_train_loader, get_eval_loader
    from wilds.common.grouper import CombinatorialGrouper
except ModuleNotFoundError:
    print('WILDS library/dependencies not found -- please install following https://github.com/p-lambda/wilds.')


D_OUTS = {"camelyon17": 2, "amazon": 5, "civilcomments": 2, "poverty": 1, "fmow": 62}

MODEL_URL = 'https://worksheets.codalab.org/rest/bundles/%s/contents/blob/best_model.pth'

N_SEEDS = {"camelyon17": 10, "amazon": 3, "civilcomments": 5, "poverty": 5, "fmow": 3}

POVERTY_FOLDS = ['A', 'B', 'C', 'D', 'E']

ALGORITHMS = ['ERM', 'IRM', 'deepCORAL', 'groupDRO', 'ERM']

MODEL_UUIDS = {
    'camelyon17': {
        'camelyon17_erm_densenet121_seed0': '0x6029addd6f714167a4d34fb5351347c6',
        'camelyon17_erm_densenet121_seed1': '0xb701f5de96064c0fa1771418da5df499',
        'camelyon17_erm_densenet121_seed2': '0x2ce5ec845b07488fb3396ab1ab8e3e17',
        'camelyon17_erm_densenet121_seed3': '0x70f110e8a86e4c3aa2688bc1267e6631',
        'camelyon17_erm_densenet121_seed4': '0x0fe16428860749d6b94dfb1fe9ffe986',
        'camelyon17_erm_densenet121_seed5': '0x0dc383dbf97a491fab9fb630c4119e3d',
        'camelyon17_erm_densenet121_seed6': '0xb7884cbe61584e80bfadd160e1514570',
        'camelyon17_erm_densenet121_seed7': '0x6f1aaa4697944b24af06db6a734f341e',
        'camelyon17_erm_densenet121_seed8': '0x043be722cf50447d9b52d3afd5e55716',
        'camelyon17_erm_densenet121_seed9': '0xc3ce3f5a89f84a84a1ef9a6a4a398109',
        'camelyon17_irm_densenet121_seed0': '0xa63359a5bb1c449085f611f5940278d1',
        'camelyon17_irm_densenet121_seed1': '0x71f860528a8b45b6bd0f0aa26906e6fc',
        'camelyon17_irm_densenet121_seed2': '0x8184a0b3a1d54cf895ce4d36db9110d0',
        'camelyon17_irm_densenet121_seed3': '0xc5fd2d287a6c4f94a424e4025cd03d3f',
        'camelyon17_irm_densenet121_seed4': '0xc1e5f84c7a05476fbcc9ebe98614e110',
        'camelyon17_irm_densenet121_seed5': '0x29c4a95f9ca644f481de41aa167c8830',
        'camelyon17_irm_densenet121_seed6': '0x02c51a59e380417ba516a3b56688c4d3',
        'camelyon17_irm_densenet121_seed7': '0x5e6bfa1e641d4ecd99de2361290209d3',
        'camelyon17_irm_densenet121_seed8': '0x1a0ac11aaeeb4a9495c37b6ab06331c9',
        'camelyon17_irm_densenet121_seed9': '0x0ce8a0a5c8be4da7ad47b1120554c62d',
        'camelyon17_deepCORAL_densenet121_seed0': '0x7966e810326842deb2377bf5f36fb60d',
        'camelyon17_deepCORAL_densenet121_seed1': '0x9d9caa8232d846c3a7ca30718e232157',
        'camelyon17_deepCORAL_densenet121_seed2': '0x8b901447f8714621b1047423844ecd37',
        'camelyon17_deepCORAL_densenet121_seed3': '0xa8f8a5bad2cc4514afe06b997f9fd648',
        'camelyon17_deepCORAL_densenet121_seed4': '0xecb7f3748c9e4640a5a0b47b54977a24',
        'camelyon17_deepCORAL_densenet121_seed5': '0xee62fda4353b42a48a127d374d0f1613',
        'camelyon17_deepCORAL_densenet121_seed6': '0x98bffe597d264f06af4ca817a01c53fa',
        'camelyon17_deepCORAL_densenet121_seed7': '0x621af9d733234b6db1de187425b8457e',
        'camelyon17_deepCORAL_densenet121_seed8': '0x717afb8719b141a8adeeb634ecbed1a3',
        'camelyon17_deepCORAL_densenet121_seed9': '0xa7731d1d205e4b51a545e75768fe7ea1',
        'camelyon17_groupDRO_densenet121_seed0': '0x583b462eaef54d93ac03f50d210d0adf',
        'camelyon17_groupDRO_densenet121_seed1': '0x296560e13e60464e9bbd8b637df21594',
        'camelyon17_groupDRO_densenet121_seed2': '0xd13b972ca6c442d5961292518ad1e89a',
        'camelyon17_groupDRO_densenet121_seed3': '0x4b031eeb625f47e09b03a801b6fe90d9',
        'camelyon17_groupDRO_densenet121_seed4': '0x8ea2d8ba9f514e56ab6030ffc07c2735',
        'camelyon17_groupDRO_densenet121_seed5': '0x72e318b10a9f4fdf974f775b453ccb58',
        'camelyon17_groupDRO_densenet121_seed6': '0xeea6090106c9458eab1c3aa91e5db63b',
        'camelyon17_groupDRO_densenet121_seed7': '0x5b62bff7317249bdb91df2137cf5f6f0',
        'camelyon17_groupDRO_densenet121_seed8': '0x64852365891f499c946461e842a7b5dc',
        'camelyon17_groupDRO_densenet121_seed9': '0x2659f54957e144809b6f4f5ffe6ddbfb',
        'camelyon17_erm_ID_seed0': '0xa46957fa425f4168a9e6fbfa500d2d4f',
        'camelyon17_erm_ID_seed1': '0x3abd16ec8af7498d9ea1ff63175b5c76',
        'camelyon17_erm_ID_seed2': '0x8119b73f481a4b3c904f227a1305fb88',
        'camelyon17_erm_ID_seed3': '0x480d53a5654543a39fe2ae9296c30304',
        'camelyon17_erm_ID_seed4': '0xec0885b489ad4bc2bc7b8958b08af824',
        'camelyon17_erm_ID_seed5': '0x8b9d4c81b59149b7a3c50a255d0d3a6b',
        'camelyon17_erm_ID_seed6': '0x017048173dd74e9cb47779f3d0534024',
        'camelyon17_erm_ID_seed7': '0x1234db7106d24f94982d176b14c86d1c',
        'camelyon17_erm_ID_seed8': '0xedf8885660a647f49a324b6cece94f15',
        'camelyon17_erm_ID_seed9': '0x90884c39ef114925bd78d6c2e7d1acc3',
    },
    'civilcomments': {
        'civilcomments_distilbert_erm_seed0': '0x17807ae09e364ec3b2680d71ca3d9623',
        'civilcomments_distilbert_erm_seed1': '0x0f6f161391c749beb1d0006238e145d0',
        'civilcomments_distilbert_erm_seed2': '0xb92f899d126d4c6ba73f2730d76ca3e6',
        'civilcomments_distilbert_erm_seed3': '0x090f8d901fad4bd7be5adb4f30e20271',
        'civilcomments_distilbert_erm_seed4': '0x7a2e24652b8d4129bc67368864062bb4',
        'civilcomments_distilbert_irm_seed0': '0x107e65f8c89642bcabe7628221dfa108',
        'civilcomments_distilbert_irm_seed1': '0x6e46b06afff04441940d967126e4a353',
        'civilcomments_distilbert_irm_seed2': '0x45db8e5cbec54c078c9dfb24cb907669',
        'civilcomments_distilbert_irm_seed3': '0x84bb3f7240484b0abfc08f9c85abefeb',
        'civilcomments_distilbert_irm_seed4': '0x7c572477e4bc4aa38679e8409a0504f9',
        'civilcomments_distilbert_deepcoral_seed0': '0x272bffce865c42c5aad565c84fbaefdc',
        'civilcomments_distilbert_deepcoral_seed1': '0xc04f7ffc47bc4544b552ddff0fcf2b5e',
        'civilcomments_distilbert_deepcoral_seed2': '0x24faf0290c174d2e8be0048fd39de6a0',
        'civilcomments_distilbert_deepcoral_seed3': '0x4681a19b29a6443a91bdc5dcc4c2047d',
        'civilcomments_distilbert_deepcoral_seed4': '0x6d282a0b9f4e415bad269947f9d59710',
        'civilcomments_distilbert_groupDRO_groupby-black-y_seed0': '0x3aeeb77983a444878cb75d7f642d6159',
        'civilcomments_distilbert_groupDRO_groupby-black-y_seed1': '0x49a1ef33666f43998f46c5a1b5e6afc9',
        'civilcomments_distilbert_groupDRO_groupby-black-y_seed2': '0xe5394cf75f4b4933b6527f51816f839c',
        'civilcomments_distilbert_groupDRO_groupby-black-y_seed3': '0xc4385ed8a1e54a8c9fd6b4a1dd8130ab',
        'civilcomments_distilbert_groupDRO_groupby-black-y_seed4': '0x99e2678b77f1479f88f2256af909f0cc',
        'civilcomments_distilbert_erm_groupby-black-y_seed0': '0x87dbe66862a74a88a718a7c77399437e',
        'civilcomments_distilbert_erm_groupby-black-y_seed1': '0x12cc6150c17d41b299d01e20cf7e9604',
        'civilcomments_distilbert_erm_groupby-black-y_seed2': '0x41e52f9e8a43440fb8071a50dfda581d',
        'civilcomments_distilbert_erm_groupby-black-y_seed3': '0xa90fa233330a42618ef4f2ee21238d3e',
        'civilcomments_distilbert_erm_groupby-black-y_seed4': '0xb5c821486b5e43bfaa479852d4a09ac7',
    },
    'fmow': {
        'fmow_erm_seed0': '0x63a3f824ac6745ea8e9061f736671304',
        'fmow_erm_seed1': '0x2f8b1417709b4f2b8eec5ead67aa6203',
        'fmow_erm_seed2': '0x8d7b4a78f9ba41b1a33c939e8280a156',
        'fmow_irm_seed0': '0x86c1b425c76348f6972279c53862ead3',
        'fmow_irm_seed1': '0xef80dd52c22a4fadb8f27827f2c0cc8e',
        'fmow_irm_seed2': '0x5f855ccf76674e76bc9e0b17e97eccc4',
        'fmow_deepcoral_seed0': '0x84da443d129a4fafa6b0485c60b2a125',
        'fmow_deepcoral_seed1': '0x8f4651313f97465f8022f628daef9044',
        'fmow_deepcoral_seed2': '0x1ef46a680072402b93186eaeb7bd8d55',
        'fmow_groupDRO_seed0': '0xb9fcbbeaf44b4dc2b8c2870ef3b06c1e',
        'fmow_groupDRO_seed1': '0xee70a2a62f5643d4b93054c42cb249d0',
        'fmow_groupDRO_seed2': '0xafdf99b8f1d74685aa01adef79291fa6',
        'fmow_erm_ID_seed0': '0x6836d02f9738458e95d0c320ee9282c4',
        'fmow_erm_ID_seed1': '0x5b6ec2f1be7a4c76873b1db7db6887a4',
        'fmow_erm_ID_seed2': '0x6f524be080bd4bee92152bec8d603444',
    },
    'poverty': {
        'poverty_erm_foldA': '0xed9774bc15d14a31be7e57517989f8b7',
        'poverty_erm_foldB': '0x30c0de563b694cc58e01d8abb48aa276',
        'poverty_erm_foldC': '0xfc22dbce36be44fe80bddaed4ffb3ff4',
        'poverty_erm_foldD': '0xcb986b1511e54a64bbb14f06be2e17a6',
        'poverty_erm_foldE': '0xdd34b17f9b8b4ea2aa4d9f72ed8573f0',
        'poverty_irm_foldA': '0xd0f659eda42f4da4a297667ae2e51b11',
        'poverty_irm_foldB': '0xa22d1c64fe9244058a58ba3853106929',
        'poverty_irm_foldC': '0xd0f659eda42f4da4a297667ae2e51b11',
        'poverty_irm_foldD': '0xd0f659eda42f4da4a297667ae2e51b11',
        'poverty_irm_foldE': '0xd0f659eda42f4da4a297667ae2e51b11',
        'poverty_deepCORAL_foldA': '0x5b4458ef8b8f4bebbc75dc7f9d84b315',
        'poverty_deepCORAL_foldB': '0xa48a6f8a725340919884389c6a1529d0',
        'poverty_deepCORAL_foldC': '0x4159f07dc87c4640aa5d66aedc12f6c4',
        'poverty_deepCORAL_foldD': '0x5e993f29628e453282c00523c59b9c11',
        'poverty_deepCORAL_foldE': '0xe8374f3986f24fbda370d91593172204',
        'poverty_groupDRO_foldA': '0x3f51b739d71440a6816ad4bd3522c7fc',
        'poverty_groupDRO_foldB': '0x4b2c90d800a544c998397ca4a5594b16',
        'poverty_groupDRO_foldC': '0x0adb692f84cc4200969d6b67b8130bb2',
        'poverty_groupDRO_foldD': '0x84c5135b6212436d96eec3d4b6f09812',
        'poverty_groupDRO_foldE': '0x795ec87eb5d848ec812112f0d18afc69',
        'poverty_erm_ID_foldA': '0x89926224750f4d0193acb898c277433b',
        'poverty_erm_ID_foldB': '0xeaea01c922bd4d8f85795759b28c6284',
        'poverty_erm_ID_foldC': '0x4f28a08a0bc14b649a15ef9d4c854e2a',
        'poverty_erm_ID_foldD': '0x3f53ee543ae3497c8fe25e8df245b3b7',
        'poverty_erm_ID_foldE': '0xe93ce4c37ea94f60a6a2ddb96360883a',
    },
    'amazon': {
        'amazonv2.0_erm_seed0': '0xe9fe4a12856f461193018504f8f65977',
        'amazonv2.0_erm_seed1': '0xcbcb1b4c49c0486eacfb082ca22b8691',
        'amazonv2.0_erm_seed2': '0xdf5298063529413eaf06654a5f83e4db',
        'amazonv2.0_irm_seed0': '0x9dd41bedfca6410880f84d857303203d',
        'amazonv2.0_irm_seed1': '0x16ab66a0c17e415cb9661779eac64ce2',
        'amazonv2.0_irm_seed2': '0x204d0f8cf55348f4b9b89767a7b1aa21',
        'amazonv2.0_deepcoral_seed0': '0x83232062e07046a999350bfa3d1ad90f',
        'amazonv2.0_deepcoral_seed1': '0x12083bbc081549fd9e943b3f0505bda6',
        'amazonv2.0_deepcoral_seed2': '0x2b3d9ccbbac3406cac4ec12d0370be8e',
        'amazonv2.0_groupDRO_seed0': '0x55e1f00c8c084c07884459331cbc1f3d',
        'amazonv2.0_groupDRO_seed1': '0x94d184bfb931478a8da909d73ed7be71',
        'amazonv2.0_groupDRO_seed2': '0x049d643074314f37845f714e6b07616a',
        'amazonv2.0_reweighted_seed0': '0xe8079c938aeb48afa57b4331e8560f38',
        'amazonv2.0_reweighted_seed1': '0x8c52c3aea8104d39a2ec505517909430',
        'amazonv2.0_reweighted_seed2': '0x68ae284c380a4f529602266ce6a8867f',
    },
}

DATASET_SPLITS = {
    'camelyon17': ['train', 'id_val', 'val', 'test'],
    'civilcomments': ['train', 'val', 'test'],
    'fmow': ['train', 'id_val', 'id_test', 'val', 'test'],
    'poverty': ['train', 'id_val', 'id_test', 'val', 'test'],
    'amazon': ['train', 'id_val', 'id_test', 'val', 'test'],
}

AMAZON_MODELS = [f'amazon_seed:{seed}_epoch:best_model.pth' for seed in range(N_SEEDS["amazon"])]
FMOW_MODELS = [f'fmow_seed:{seed}_epoch:best_model.pth' for seed in range(N_SEEDS["fmow"])]
POVERTY_MODELS = [f'poverty_fold:{fold}_epoch:best_model.pth' for fold in POVERTY_FOLDS]


class ProperDataLoader:
    """ This class defines an iterator that wraps a PyTorch DataLoader 
        to only return the first two of three elements of the data tuples.

        This is used to make the data loaders from the WILDS benchmark
        (which return (X, y, metadata) tuples, where metadata for example
        contains domain information) compatible with the uq.py script and
        with the laplace library (which both expect (X, y) tuples).
    """
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.dataset = self.data_loader.dataset

    def __iter__(self):
        self.data_iter = iter(self.data_loader)
        return self

    def __next__(self):
        X, y, _ = next(self.data_iter)
        return X, y

    def __len__(self):
        return len(self.data_loader)


def load_pretrained_wilds_model(dataset, model_dir, device, model_idx=0, model_seed=0):
    """ load pre-trained model """

    # load default config and instantiate model
    config = get_default_config(dataset, algorithm=ALGORITHMS[model_idx])
    is_featurizer = dataset in ["civilcomments", "amazon"] and ALGORITHMS[model_idx] == "deepCORAL"
    model = initialize_model(config, D_OUTS[dataset], is_featurizer=is_featurizer)
    if is_featurizer:
        model = nn.Sequential(*model)
    model = model.to(device)

    # define path to pre-trained model parameters
    model_list_idx = model_idx * N_SEEDS[dataset] + model_seed
    model_name = list(MODEL_UUIDS[dataset].keys())[model_list_idx]
    model_path = Path(model_dir) / dataset / f"{model_name}.pth"

    # if required, download pre-trained model parameters
    if not model_path.exists():
        model_path.parent.mkdir(exist_ok=True)
        model_url = MODEL_URL % MODEL_UUIDS[dataset][model_name]

        # handle special naming cases
        if dataset == "amazon":
            model_url = model_url.replace("best_model.pth", AMAZON_MODELS[model_seed])
        elif dataset == "fmow" and model_idx == 4:
            model_url = model_url.replace("best_model.pth", FMOW_MODELS[model_seed])
        elif dataset == "poverty" and model_idx == 4:
            model_url = model_url.replace("best_model.pth", POVERTY_MODELS[model_seed])

        print(f"Downloading pre-trained model parameters for {model_name} from {model_url}...")
        urllib.request.urlretrieve(model_url, model_path)

    # load pre-trained parameters into model
    print(f"Loading pre-trained model parameters for {model_name}...")
    state_dict = torch.load(model_path)["algorithm"]
    model_state_dict_keys = list(model.state_dict().keys())

    model_state_dict = {}
    for m in state_dict:
        if dataset in ["civilcomments", "amazon"] and ALGORITHMS[model_idx] == "deepCORAL" and "featurizer" in m:
            continue

        m_new = m if m.split('.')[0] == "classifier" else '.'.join(m.split('.')[1:])

        if "classifier" in m_new:
            if dataset == "poverty":
                m_new = m_new.replace("classifier", "fc")
            elif dataset in ["civilcomments", "amazon"] and ALGORITHMS[model_idx] == "deepCORAL":
                m_new = m_new.replace("classifier", "1")

        if m_new not in model_state_dict_keys:
            continue

        model_state_dict[m_new] = state_dict[m]

    model.load_state_dict(model_state_dict)
    model.eval()

    return model


def get_wilds_loaders(dataset, data_dir, data_fraction=1.0, model_seed=0):
    """ load in-distribution datasets and return data loaders """

    # load default config and the full dataset
    config = get_default_config(dataset, data_fraction=data_fraction)
    dataset_kwargs = {'fold': POVERTY_FOLDS[model_seed]} if dataset == "poverty" else {}
    full_dataset = get_dataset(dataset=dataset, root_dir=data_dir, **dataset_kwargs)
    train_grouper = CombinatorialGrouper(dataset=full_dataset, groupby_fields=config.groupby_fields)

    if dataset == "fmow":
        config.batch_size = config.batch_size // 2

    # get the train data loader
    train_transform = initialize_transform(transform_name=config.train_transform, config=config, dataset=full_dataset)
    train_data = full_dataset.get_subset('train', frac=config.frac, transform=train_transform)
    train_loader = get_train_loader(loader=config.train_loader, dataset=train_data, batch_size=config.batch_size,
                                    uniform_over_groups=config.uniform_over_groups, grouper=train_grouper,
                                    distinct_groups=config.distinct_groups, n_groups_per_batch=config.n_groups_per_batch,
                                    **config.loader_kwargs)

    # get the in-distribution validation data loader
    eval_transform = initialize_transform(transform_name=config.eval_transform, config=config, dataset=full_dataset)
    try:
        val_str = "val" if dataset == "fmow" else "id_val"
        val_data = full_dataset.get_subset(val_str, frac=config.frac, transform=eval_transform)
        val_loader = get_eval_loader(loader=config.eval_loader, dataset=val_data, batch_size=config.batch_size, grouper=train_grouper, **config.loader_kwargs)
    except:
        print(f"{dataset} dataset doesn't have an in-distribution validation split -- using train split instead!")
        val_loader = train_loader

    # get the in-distribution test data loader
    try:
        in_test_data = full_dataset.get_subset('id_test', frac=config.frac, transform=eval_transform)
        in_test_loader = get_eval_loader(loader=config.eval_loader, dataset=in_test_data, batch_size=config.batch_size, grouper=train_grouper, **config.loader_kwargs)
    except:
        print(f"{dataset} dataset doesn't have an in-distribution test split -- using validation split instead!")
        in_test_loader = val_loader

    # wrap data loaders for compatibility with uq.py and laplace library
    train_loader = ProperDataLoader(train_loader)
    val_loader = ProperDataLoader(val_loader)
    in_test_loader = ProperDataLoader(in_test_loader)

    return train_loader, val_loader, in_test_loader


def get_wilds_ood_test_loader(dataset, data_dir, data_fraction=1.0, model_seed=0):
    """ load out-of-distribution test data and return data loader """

    # load default config and the full dataset
    config = get_default_config(dataset, data_fraction=data_fraction)
    dataset_kwargs = {'fold': POVERTY_FOLDS[model_seed]} if dataset == "poverty" else {}
    full_dataset = get_dataset(dataset=dataset, root_dir=data_dir, **dataset_kwargs)
    train_grouper = CombinatorialGrouper(dataset=full_dataset, groupby_fields=config.groupby_fields)

    if dataset == "fmow":
        config.batch_size = config.batch_size // 2

    # get the OOD test data loader
    test_transform = initialize_transform(transform_name=config.eval_transform, config=config, dataset=full_dataset)
    test_data = full_dataset.get_subset('test', frac=config.frac, transform=test_transform)
    test_loader = get_eval_loader(loader=config.eval_loader, dataset=test_data, batch_size=config.batch_size, grouper=train_grouper, **config.loader_kwargs)

    # wrap data loader for compatibility with uq.py and laplace library
    test_loader = ProperDataLoader(test_loader)

    return test_loader


def get_default_config(dataset, algorithm="ERM", data_fraction=1.0):
    config = Namespace(dataset=dataset, algorithm=algorithm, model_kwargs={}, optimizer_kwargs={},
                        loader_kwargs={}, dataset_kwargs={}, scheduler_kwargs={}, 
                        train_transform=None, eval_transform=None, no_group_logging=True, 
                        distinct_groups=True, frac=data_fraction, scheduler=None)
    return populate_defaults(config)


def optimize_noise_standard_deviation(model, val_loader, device, lr=1e-1, n_epochs=10):
    """ optimizes the noise standard deviation of a Gaussian regression likelihood on the validation data """

    # define parameter to optimize and optimizer
    log_sigma_noise = nn.Parameter(torch.zeros(1, device=device))
    optimizer = torch.optim.Adam([log_sigma_noise], lr=lr)

    # define Gaussian negative log-likelihood loss; this is equivalent to
    # lambda y_pred, y, var: -Normal(y_pred, var.sqrt()).log_prob(y).mean(dim=0)
    gaussian_nll_loss = nn.GaussianNLLLoss(full=True)

    for e in range(n_epochs):
        print(f"Running epoch {e+1}/{n_epochs}...")
        for i, (X, y) in enumerate(val_loader):
            optimizer.zero_grad()
            y_pred = model(X.to(device))
            if isinstance(y_pred, tuple):
                y_pred = y_pred[0]
            nll = gaussian_nll_loss(y_pred, y.to(device), torch.ones_like(y_pred) * log_sigma_noise.exp()**2)
            nll.backward()
            optimizer.step()
            sigma_noise = log_sigma_noise.exp().item()
            print(f"\tIter {i+1}/{len(val_loader)}: sigma_noise = {sigma_noise} (NLL: {nll.item()}).")
        print(f"After epoch {e+1}/{n_epochs}: sigma_noise = {sigma_noise}.\n")

    return sigma_noise
