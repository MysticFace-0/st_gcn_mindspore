import mindspore
import torch

x = torch.rand([1,2,3,4,5])
x = x.permute(0, 1, 3, 4, 2)
print(x.shape)
