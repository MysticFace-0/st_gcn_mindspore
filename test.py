import mindspore
import torch
from mindspore import ops

x = torch.Tensor([[[1,2,3],[3,4,5]],[[4,5,6],[6,5,4]]])
print(x.size())
y = torch.Tensor([[[1,2],[2,1],[1,2]],[[4,5],[5,4],[4,5]]])
print(y.size())
print(torch.matmul(x,y).size(),torch.matmul(x,y))

matmul = ops.MatMul()
x=mindspore.Tensor([[[1,2,3],[3,4,5]],[[4,5,6],[6,5,4]]])
y=mindspore.Tensor([[[1,2],[2,1],[1,2]],[[4,5],[5,4],[4,5]]])
bx, hx, wx = x.shape
by, hy, wy = y.shape
ans = matmul(x[0],y[0])
for i in range(1,bx):
    ans = ops.concat((ans, matmul(x[i],y[i])),0)
ans = ans.view(bx,hx,wy)
print(ans.shape, ans)

