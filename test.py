import torch
from torch import nn

data_bn = nn.BatchNorm1d(3 * 17)

x = torch.rand(2*1, 1,100, 17, 3)
N, M, T, V, C = x.size()
x = x.permute(0, 1, 3, 4, 2).contiguous()

x = data_bn(x.view(N * M, V * C, T))
x = x.view(N, M, V, C, T).permute(0, 1, 3, 4,
                                  2).contiguous().view(N * M, C, T, V)

print(x.shape)