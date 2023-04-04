import math
import numpy as np

import mindspore
import mindspore.nn as nn
from mindspore import Parameter, Tensor, ops
from mindspore.common.initializer import initializer, HeNormal

from torch.autograd import Variable

from utils.graph import Graph


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


class unit_tcn(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, pad, 0, 0),
                              pad_mode= "pad", stride=(stride, 1), has_bias=True)

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def construct(self, x):
        x = self.bn(self.conv(x))
        return x


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


class TCN_GCN_unit(nn.Cell):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A)
        self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def construct(self, x):
        x = self.tcn1(self.gcn1(x)) + self.residual(x)
        return self.relu(x)


class AGCN(nn.Cell):
    def __init__(self, num_class=60, num_point=17, num_person=1, num_frames=100, graph_args=dict(), in_channels=3):
        super(AGCN, self).__init__()

        self.graph = Graph(**graph_args)
        A = Parameter(Tensor(self.graph.A, dtype=mindspore.float32), requires_grad=False)

        self.data_bnt = nn.BatchNorm1d(num_frames)
        self.data_bnc = nn.BatchNorm1d(num_person  * A.shape[1] * in_channels) # M*V*C
        # mindspore无三维算子，先要在L上归一化，再在C上归一化

        self.l1 = TCN_GCN_unit(3, 64, A, residual=False)
        self.l2 = TCN_GCN_unit(64, 64, A)
        self.l3 = TCN_GCN_unit(64, 64, A)
        self.l4 = TCN_GCN_unit(64, 64, A)
        self.l5 = TCN_GCN_unit(64, 128, A, stride=2)
        self.l6 = TCN_GCN_unit(128, 128, A)
        self.l7 = TCN_GCN_unit(128, 128, A)
        self.l8 = TCN_GCN_unit(128, 256, A, stride=2)
        self.l9 = TCN_GCN_unit(256, 256, A)
        self.l10 = TCN_GCN_unit(256, 256, A)

        self.fc = nn.Dense(256, num_class)
        constant_init_weight = mindspore.common.initializer.Normal(sigma=math.sqrt(2. / num_class), mean=0.0)
        constant_init_weight(self.fc.weight)
        bn_init(self.data_bn, 1)

    def construct(self, x):
        # B batch_size
        # N 视频个数
        # C = 3(X, Y, S) 代表一个点的信息(位置 + 预测的可能性)
        # T = 100 一个视频的帧数paper规定是100帧，不足的重头循环，多的clip
        # V 17 根据不同的skeleton获得的节点数而定
        # M = 1 人数，paper中将人数限定在最大1个人

        # B, N, M, T, V, C to N, C, T, V, M
        B, N, M, T, V, C = x.shape
        x = x.view(B*N, M, T, V, C)
        x = x.transpose(0,4,2,3,1)

        N, C, T, V, M = x.shape

        x = x.permute(0, 4, 3, 1, 2).view(N, M * V * C, T)

        # 先对T做归一化
        x = self.data_bnt(x.view(N * M * V * C, T)).view(N, M * V * C, T)
        x = x.transpose((0, 2, 1))
        # 再对C做归一化
        x = self.data_bnc(x.view(N * T, M * V * C)).view(N, T, M * V * C)
        x = x.transpose((0, 2, 1))

        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).view(N * M, C, T, V)

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

        # N*M,C,T,V
        c_new = x.shape[1]
        x = x.view(N, M, c_new, -1)
        x = ops.mean(x, 3, keep_dims=False)
        x = ops.mean(x, 1, keep_dims=False)

        x=self.fc(x)

        return x

if __name__=="__main__":
    # model测试
    model = AGCN(num_class = 60, num_point = 17, num_person = 1, num_frames=100, graph_args = dict(layout='coco', mode='spatial'), in_channels = 3,)
    shape = (2, 1, 1, 100, 17, 3) # dataloader直接读取的格式
    uniformreal = mindspore.ops.UniformReal(seed=2)
    x = uniformreal(shape)
    y = model(x)
    print(y.shape)
