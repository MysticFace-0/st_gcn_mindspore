import numpy as np
import torch
from mindspore import Tensor, ParameterTuple, ops, Parameter
from torch import nn
import mindspore


class Block(mindspore.nn.Cell):
  def __init__(self, C,L):
      super().__init__()
      self.data_bnl = mindspore.nn.BatchNorm1d(L)
      self.data_bnn = mindspore.nn.BatchNorm1d(C)
  def construct(self, x):
      N, C, L = x.shape
      x = self.data_bnl(x.view(N * C, L)).view(N, C, L)
      x = x.transpose((0, 2, 1))
      # 再对C做归一化
      x = self.data_bnn(x.view(N * L, C)).view(N, L, C)
      x = x.transpose((0, 2, 1))
      return x

x = mindspore.Tensor([[[1.,2.],[3.,-2.],[-1.,-2.],[-3.,2.]],[[-1.,-2.],[-3.,2.],[1.,2.],[3.,-2.]]],mindspore.float32)#2,3,4
N, C, L = x.shape
block = Block(C, L)
block.set_train()
y = block(x)
print(y)

