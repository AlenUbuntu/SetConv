from .dataset import StandardDataset
from .sampler import SetSampler
import pickle
import torch
import config.paths_catalog as paths_catalog
import pandas as pd


D = {
    'StandardDataset': StandardDataset
}

def build_dataset(cfg, name, dataset_catalog, ratio=None, is_train=True, is_valid=False):
    """
    Arguments:
        name (str): name of the dataset
        dataset_catalog (DatasetCatalog): contains the information on how to construct a dataset.
        ratio (float): train/test ratio
        is_train (bool): whether to setup the dataset for training or testing
    """
    assert (is_train and is_valid) == False
    data_config = dataset_catalog.get(cfg, name, ratio)
    factory = D[data_config['factory']]
    # load the dataset
    data = pd.read_csv(data_config['path'], header=None)
    x, y = data.iloc[:, :-1].values, data.iloc[:, -1].values
    if is_train:
        index_train = pd.read_csv(data_config['index_train'])
        index_train = index_train.values.reshape(-1)
        x, y = x[index_train], y[index_train]
    elif is_valid:
        index_valid = pd.read_csv(data_config['index_valid'])
        index_valid = index_valid.values.reshape(-1)
        x, y = x[index_valid], y[index_valid]
    else:
        index_test = pd.read_csv(data_config['index_test'])
        index_test = index_test.values.reshape(-1)
        x, y = x[index_test], y[index_test]

    dataset = factory(cfg, x, y)

    return dataset 

def load_dataset(cfg, train=True, valid=False):
    assert (train and valid) == False
    DatasetCatalog = paths_catalog.DatasetCatalog
    if train:
        data_config = DatasetCatalog.get(cfg, cfg.DATASET.TRAIN, cfg.DATASET.RATIO)
        # load the dataset
        data = pd.read_csv(data_config['path'], header=None)
        x, y = data.iloc[:, :-1].values, data.iloc[:, -1].values
        index_train = pd.read_csv(data_config['index_train'])
        index_train = index_train.values.reshape(-1)
        x, y = x[index_train], y[index_train]
    elif valid:
        data_config = DatasetCatalog.get(cfg, cfg.DATASET.VALID, cfg.DATASET.RATIO)
        # load the dataset
        data = pd.read_csv(data_config['path'], header=None)
        x, y = data.iloc[:, :-1].values, data.iloc[:, -1].values
        index_valid = pd.read_csv(data_config['index_valid'])
        index_valid = index_valid.values.reshape(-1)
        x, y = x[index_valid], y[index_valid]
    else:
        data_config = DatasetCatalog.get(cfg, cfg.DATASET.TEST, cfg.DATASET.RATIO)
        # load the dataset
        data = pd.read_csv(data_config['path'], header=None)
        x, y = data.iloc[:, :-1].values, data.iloc[:, -1].values
        index_test = pd.read_csv(data_config['index_test'])
        index_test = index_test.values.reshape(-1)
        x, y = x[index_test], y[index_test]
    return x, y



def make_data_sampler(dataset, shuffle):
    if shuffle:
        return None  # we utilize customized batch sampler to maintain the imbalance ratio, which is mutually exclusive with sampler
    else:
        return torch.utils.data.sampler.SequentialSampler(dataset)

def make_batch_sampler(cfg, dataset, is_train=True):
    if is_train:
        # utilize set sampler
        batch_sampler = SetSampler(dataset, cfg.DATALOADER.NUM_BATCH, cfg.DATALOADER.BATCH_SIZE)
    else:
        batch_sampler = None  # use the default batch sampler 

    return batch_sampler 


def make_data_loader(cfg, is_train=True, is_valid=False):
    assert (is_train and is_valid) == False 

    if is_train:
        shuffle = True 
    else:
        shuffle = False
    
    DatasetCatalog = paths_catalog.DatasetCatalog
    if is_train:
        dataset = build_dataset(cfg, cfg.DATASET.TRAIN, DatasetCatalog, ratio=cfg.DATASET.RATIO, is_train=is_train, is_valid=is_valid)
    elif is_valid:
        dataset = build_dataset(cfg, cfg.DATASET.VALID, DatasetCatalog, ratio=cfg.DATASET.RATIO, is_train=is_train, is_valid=is_valid)
    else:
        dataset = build_dataset(cfg, cfg.DATASET.TEST, DatasetCatalog, ratio=cfg.DATASET.RATIO, is_train=is_train, is_valid=is_valid)

    # data sampler
    data_sampler = make_data_sampler(dataset, shuffle=shuffle)
    batch_sampler = make_batch_sampler(cfg, dataset, is_train=is_train)

    data_loader = torch.utils.data.DataLoader(
        dataset, 
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        sampler=data_sampler,
        batch_sampler=batch_sampler
    )

    return data_loader
