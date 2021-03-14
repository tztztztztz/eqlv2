import torch
from torch import nn
from torch.nn import functional as F


class LWSLinear(nn.Linear):
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True):
        super(nn.Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.weight_scaler = nn.Parameter(torch.Tensor(out_features, 1))
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        nn.init.constant_(self.weight_scaler, 1.0)

    def forward(self, input):
        return F.linear(input, self.weight * self.weight_scaler, self.bias)