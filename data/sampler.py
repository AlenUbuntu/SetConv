from torch.utils.data import Sampler
from torch._six import int_classes as _int_classes
import numpy as np
import torch  


class SetSampler(Sampler):
    def __init__(self, dataset, n_batch, batch_size):
        if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
                    
        if not isinstance(n_batch, _int_classes) or isinstance(n_batch, bool) or n_batch <= 0:
            raise ValueError("n_batch should be a positive integer value, "
                             "but got n_batch={}".format(n_batch))

        self.batch_size = batch_size
        self.dataset = dataset 
        self.n_batch = n_batch

        # compute sample size for each class in a mini-batch
        sample_size = {}
        remain_size = self.batch_size
        for c, ratio in self.dataset.clsratio:
            size = min(max(int(self.batch_size * ratio)+1, 1), remain_size)
            remain_size -= size
            sample_size[c] = size
        self.sample_size = sample_size

    
    def __len__(self,):
        return self.n_batch

    def __iter__(self):
        for i in range(self.n_batch):
            batch = []

            for c in self.sample_size:
                l = self.dataset.cls2idx[c]
                pos = torch.randperm(len(l))[:self.sample_size[c]]
                if isinstance(l[pos], (list, np.ndarray, tuple)):
                    batch.append(torch.from_numpy(l[pos]).view(-1, 1))
                else:
                    batch.append(torch.from_numpy(np.array([l[pos]])).view(-1, 1))
            batch = torch.cat(batch, dim=0)
            yield batch
            
            