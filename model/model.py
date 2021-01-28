from .set_convolution import SetConvLayer
from data import load_dataset
import torch 
import torch.nn as nn 
from sklearn.cluster import KMeans
import numpy as np 


# binary classifier
class SetConvNetwork(torch.nn.Module):
    def __init__(self, cfg, anchor):
        super(SetConvNetwork, self).__init__()
        self.cfg = cfg 
        self.anchor = anchor

        prev_dim = cfg.MODEL.INPUT_DIM
        layers = []
        for dim in cfg.MODEL.DOWNSAMPLE_DIM:
            layers.append(nn.Linear(prev_dim, dim))
            prev_dim = dim
        self.downsample = nn.Sequential(*layers)

        # create a set convolution layer for each minority class
        # tmp = [('class {}'.format(int(c)), SetRelativeConvolution(cfg, prev_dim, self.cfg.MODEL.SETCONV_DIM)) for c in anchors]
        # self.set_conv = nn.ModuleDict(dict(tmp))
        self.set_conv = SetConvLayer(cfg, prev_dim, self.cfg.MODEL.SETCONV_DIM)

        self.relu = nn.ReLU(inplace=True)

        self.fc = nn.Linear(self.cfg.MODEL.SETCONV_DIM, self.cfg.MODEL.EMBEDDING_DIM, bias=False)


        # initialize the fc layer
        nn.init.xavier_uniform_(self.fc.weight)
        for layer in self.downsample:
            nn.init.kaiming_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        # anchors = {}
        # for c in self.anchors:
        #     anchors[c] = self.relu(self.downsample(self.anchors[c]))
        anchor = self.relu(self.downsample(self.anchor))
        x = self.downsample(x) 
        x = self.relu(x)

        # outs = []
        # for c in anchors:
        #     out = self.set_conv['class {}'.format(int(c))](x-anchors[c], x)
        #     outs.append(out)
        out = self.set_conv(x-anchor, x)

        # out = torch.cat(outs, dim=1)
        out = self.fc(out)
        out = self.relu(out)
        return out

def build_model(cfg):
    minority_classes = cfg.MODEL.MINORITY_CLASS
    x, y = load_dataset(cfg)
    # anchors = {}
    # for c in minority_classes:
    #     data = x[y==c]
    #     if len(data) >= cfg.MODEL.K:
    #         kmeans = KMeans(n_clusters=cfg.MODEL.K)
    #     else: 
    #         kmeans = KMeans(n_clusters=1)
    #     kmeans.fit(data)
    #     centers = kmeans.cluster_centers_
    #     centers = torch.from_numpy(centers).to(cfg.DEVICE)
    #     anchors[c] = centers.float()
    data = x[y==minority_classes]
    anchor = torch.from_numpy(np.mean(data, axis=0).reshape(1, -1)).float().to(cfg.DEVICE)
    model = SetConvNetwork(cfg, anchor)

    return model
