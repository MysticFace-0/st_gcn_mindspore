import math
import numpy as np

import mindspore
from mindspore import Parameter, Tensor, ops

from mindspore.common.initializer import initializer, HeNormal
import torch
from torch.autograd import Variable
import mindspore.nn as nn

from model.stgcn.utils.graph import Graph


def conv_branch_init(conv, branches):
    n, k1, k2 = conv.weight.shape
    constant_init_weight = mindspore.common.initializer.Normal(sigma=math.sqrt(2. / (n * k1 * k2 * branches)), mean=0.0)
    constant_init_bias = mindspore.common.initializer.Constant(value=0)
    constant_init_weight(conv.weight)
    constant_init_bias(conv.bias)

def conv_init(conv):
    constant_init_bias = mindspore.common.initializer.Constant(value=0)
    conv.weight = initializer(HeNormal(mode='fan_out'), conv.weight.shape, mindspore.float32)
    constant_init_bias(conv.bias)

def bn_init(bn, scale):
    constant_init_weight = mindspore.common.initializer.Constant(value=scale)
    constant_init_bias = mindspore.common.initializer.Constant(value=0)
    constant_init_weight(bn.gamma.value())
    constant_init_bias(bn.beta.value())

class unit_gcn(nn.Cell):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.PA = Parameter(A, requires_grad=True)
        constant_init_bias = mindspore.common.initializer.Constant(value=1e-6)
        constant_init_bias(self.PA)
        self.A = Parameter(A, requires_grad=False)
        self.num_subset = num_subset

        self.conv_a = nn.CellList()
        self.conv_b = nn.CellList()
        self.conv_d = nn.CellList()
        for i in range(self.num_subset):
            self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down = nn.SequentialCell(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.matmul = ops.MatMul()
        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()

        for m in self.cells():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            print(self.conv_d[i])
            conv_branch_init(self.conv_d[i], self.num_subset)

    def construct(self, x):
        N, C, T, V = x.shape
        A = self.A
        A = A + self.PA

        y = None
        for i in range(self.num_subset):
            A1 = self.conv_a[i](x).permute(0, 3, 1, 2).view(N, V, self.inter_c * T)
            A2 = self.conv_b[i](x).view(N, self.inter_c * T, V)
            A1 = self.soft(self.matmul(A1, A2) / A1.size(-1))  # N V V
            A1 = A1 + A[i]
            A2 = x.view(N, C * T, V)
            z = self.conv_d[i](self.matmul(A2, A1).view(N, C, T, V))
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        return self.relu(y)

graph = Graph(**dict(layout='coco', mode='spatial'))
A = Parameter(Tensor(graph.A, dtype=mindspore.float32), requires_grad=False)
gcn = unit_gcn(3, 64, A)
shape = (2, 3, 100, 17)
uniformreal = mindspore.ops.UniformReal(seed=2)
x = uniformreal(shape)
# y = gcn(x)
# print(y.shape)

