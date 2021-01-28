from torch.utils.data import Dataset 
import numpy as np


class StandardDataset(Dataset):
    def __init__(self, cfg, x, y):
        self.cfg = cfg 
        self.x = x 
        self.y = y 
        self.classes = set(self.y)

        idx = np.arange(len(self.y))
        cls2idx = {}
        clsratio = []
        for c in self.classes:
            cls_idx = idx[self.y==c]
            cls2idx[c] = cls_idx 
            clsratio.append((c, len(cls_idx)/len(self.y)))
        
        self.cls2idx = cls2idx
        self.clsratio = sorted(clsratio, key=lambda x: x[1])
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
