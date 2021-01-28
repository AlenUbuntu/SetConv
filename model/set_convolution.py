import torch 
import torch.nn as nn 

class SetConvLayer(torch.nn.Module):
    def __init__(self, cfg, in_dim, out_dim):
        super(SetConvLayer, self).__init__()
        self.cfg = cfg 
        self.fc = nn.Linear(in_dim, out_dim, bias=True)
        self.w = nn.Parameter(torch.ones(in_dim, out_dim), requires_grad=True)

        # initialize the weight matrix
        # since we are going to use ReLU as the activation function, we utilize kaiming initialization
        torch.nn.init.kaiming_uniform_(self.fc.weight, nonlinearity='relu')
        torch.nn.init.zeros_(self.fc.bias)
    
    
    def forward(self, x, feature):
        # x - (N, I)
        n, i = x.shape
        weight1 = self.fc(x)  # (N, O)

        # compute khatri_rao product
        n, o = weight1.shape

        weight1 = weight1.unsqueeze(1).permute(2, 0, 1)
        weight2 = torch.softmax(self.w, dim=0)
        weight2 = weight2.unsqueeze(-1).permute(1, 2, 0)
        w = torch.bmm(weight1, weight2).permute(1, 2, 0)
        out = torch.sum(w * feature.unsqueeze(-1), (0, 1)).view(1, -1)
        out /= n
        return out
