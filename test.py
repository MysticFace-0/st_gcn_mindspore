

import torch
from mindspore import Tensor, ParameterTuple, ops, Parameter
from torch import nn
import mindspore


# data_bn = nn.BatchNorm1d(2)
# x = torch.Tensor([[[1,2,3,4],[5,6,7,8],[9,10,11,12]],[[12,11,10,9],[8,7,6,5],[4,3,2,1]]])#2,3,4
# x = torch.Tensor([[[2,3,4],[3,2,1]],[[4,3,2],[1,2,3]],[[2,3,4],[3,2,1]]])#N *C L  3 2 3
# # print(x)
# x = data_bn(x)
# print(x)

# tensor([[[ 1.,  2.,  3.,  4.],
#          [ 5.,  6.,  7.,  8.],
#          [ 9., 10., 11., 12.]],
#
#         [[12., 11., 10.,  9.],
#          [ 8.,  7.,  6.,  5.],
#          [ 4.,  3.,  2.,  1.]]])
# tensor([[[-1.3242, -1.0835, -0.8427, -0.6019],
#          [-1.3416, -0.4472,  0.4472,  1.3416],
#          [ 0.6019,  0.8427,  1.0835,  1.3242]],
#
#         [[ 1.3242,  1.0835,  0.8427,  0.6019],
#          [ 1.3416,  0.4472, -0.4472, -1.3416],
#          [-0.6019, -0.8427, -1.0835, -1.3242]]],
#        grad_fn=<NativeBatchNormBackward0>)

# x = mindspore.Tensor([[[2,3,4],[3,2,1]],[[4,3,2],[1,2,3]],[[2,3,4],[3,2,1]]],mindspore.float32)#2,3,4
# N, C, L = x.shape
# data_bnl = mindspore.nn.BatchNorm1d(L)
# data_bnn = mindspore.nn.BatchNorm1d(N)
# x = data_bnl(x.view(N*C,L)).view(N, C, L)
#
# x = x.transpose((2,1,0))
# x = data_bnn(x.view(L*C,N)).view(L, C, N)
# x = x.transpose((2,1,0))
# print(x)

x = torch.rand(2, 256, 25, 17)
shape = (2, 256, 25, 17)
uniformreal = mindspore.ops.UniformReal(seed=2)
x = uniformreal(shape)
print(x.shape[2:])