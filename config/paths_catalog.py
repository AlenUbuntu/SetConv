import os 

class DatasetCatalog(object):
    DATASETS = {
        'AmzBooks': 'amz_review/books2.csv',
    }

    @staticmethod
    def get(cfg, name, ratio=None):
        if name not in DatasetCatalog.DATASETS:
            raise RuntimeError("Dataset not available: {}".format(name))

        data_dir = cfg.DATASET.DIR 
        rel_path = DatasetCatalog.DATASETS[name]
        path = os.path.join(data_dir, rel_path)
        if ratio:
            return dict(
                factory='StandardDataset',
                path=path,
                index_train=path[:-4]+'_{}_train_idx.csv'.format(ratio),
                index_valid=path[:-4]+'_{}_valid_idx.csv'.format(ratio),
                index_test=path[:-4]+'_{}_test_idx.csv'.format(ratio)
            )
        else:
            return dict(
                factory='StandardDataset',
                path=path
            )
