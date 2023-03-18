import mindspore
from mindspore import ops

from .st_gcn import Model as ST_GCN

class Model(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

        self.origin_stream = ST_GCN(*args, **kwargs)
        self.motion_stream = ST_GCN(*args, **kwargs)

    def construct(self, x):

        mindspore.set_context(device_target='GPU') #将张量存到GPU上

        N, C, T, V, M = x.shape
        # m = torch.cat((torch.cuda.FloatTensor(N, C, 1, V, M).zero_(),
        #                 x[:, :, 1:-1] - 0.5 * x[:, :, 2:] - 0.5 * x[:, :, :-2],
        #                 torch.cuda.FloatTensor(N, C, 1, V, M).zero_()), 2)
        cat_op = mindspore.ops.Concat(2) #二维拼接算子
        zeros = ops.Zeros() # 返回值为0的Tensor，其shape和数据类型与输入Tensor相同。
        m = cat_op((zeros((N, C, 1, V, M), mindspore.float32).set_context(device_target='GPU'),
                        x[:, :, 1:-1] - 0.5 * x[:, :, 2:] - 0.5 * x[:, :, :-2],
                        zeros((N, C, 1, V, M), mindspore.float32).set_context(device_target='GPU')))

        res = self.origin_stream(x) + self.motion_stream(m)
        return res