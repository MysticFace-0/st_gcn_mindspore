import numpy as np
import torch
from mindspore import Tensor, ParameterTuple, ops, Parameter
from torch import nn
import mindspore

shape1 = (4,60)  # dataloader直接读取的格式
uniformreal = mindspore.ops.UniformReal(seed=2)
y = uniformreal(shape1)
label = mindspore.Tensor([13,53,3,23],dtype=mindspore.int32)

celoss = mindspore.nn.CrossEntropyLoss()
loss = celoss(y,label)

print(loss)

