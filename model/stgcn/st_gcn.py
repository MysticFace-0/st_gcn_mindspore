import mindspore
import mindspore.nn as nn
import numpy

from mindspore import Parameter, Tensor, ParameterTuple, ops

from utils.tgcn import ConvTemporalGraphical
from utils.graph import Graph

class STGCN(nn.Cell):
    r"""Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        num_frames (int): Number of frames for the single video
        num_class (int): Number of classes for the classification task
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self, in_channels, num_frames, num_class, graph_args,
                 edge_importance_weighting, **kwargs):
        super().__init__(auto_prefix=True)

        # load graph
        self.graph = Graph(**graph_args)
        # self.A = Parameter(Tensor(self.graph.A, dtype=mindspore.float32), requires_grad=False)
        self.A = Tensor(self.graph.A, dtype=mindspore.float32)
        # build networks
        spatial_kernel_size = self.A.shape[0]
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size) # (9,3)

        # 只对C归一化
        self.data_bnc = nn.BatchNorm1d(in_channels * self.A.shape[1])
        # # 对L和C归一化
        # self.data_bnt = nn.BatchNorm1d(num_frames, affine=False, gamma_init='zeros',
        #                                beta_init='zeros', moving_mean_init='zeros', moving_var_init='zeros')
        # self.data_bnc = nn.BatchNorm1d(in_channels * self.A.shape[1])
        # # mindspore无三维算子，先要在L上归一化，再在C上归一化

        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.CellList([
            st_gcn(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 128, kernel_size, 2, **kwargs),
            st_gcn(128, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 256, kernel_size, 2, **kwargs),
            st_gcn(256, 256, kernel_size, 1, **kwargs),
            st_gcn(256, 256, kernel_size, 1, **kwargs),
        ])

        # initialize parameters for edge importance weighting
        ones = ops.Ones()
        if edge_importance_weighting:
            # warning: Please set a unique name for the parameter in ParameterTuple
            one = Parameter(ones(self.A.shape, mindspore.float32))
            self.edge_importance = ParameterTuple(
                one for i in self.st_gcn_networks
            )
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks) # [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        # fcn for prediction

        self.fcn = nn.Conv2d(256, num_class, kernel_size=1, has_bias=True)

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

        # data normalization
        N, C, T, V, M = x.shape # (2, 3, 100, 17, 1)
        x = x.permute(0, 4, 3, 1, 2)#.contiguous()
        x = x.view(N * M, V * C, T) # (2*1, 3*17, 100)

        # 只对C归一化
        x = x.transpose((0, 2, 1))
        x = self.data_bnc(x.view(N * M * T, V * C)).view(N * M, T, V * C)
        x = x.transpose((0, 2, 1))

        # # 对L和C归一化
        # # 先对T做归一化
        # x = self.data_bnt(x.view(N * M * V * C, T)).view(N * M, V * C, T)
        # x = x.transpose((0, 2, 1))
        # # 再对C做归一化
        # x = self.data_bnc(x.view(N * M * T, V * C)).view(N * M, T, V * C)
        # x = x.transpose((0, 2, 1))

        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2)#.contiguous()
        x = x.view(N * M, C, T, V) # human1_video=[0:num_clip], human2_video=[num_clip:2*num_clip]

        # forwad
        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # global pooling
        avgpool_op = ops.AvgPool(pad_mode="VALID", kernel_size=x.shape[2:], strides=1)
        x = avgpool_op(x)
        x = ops.mean(x.view(N, M, -1, 1, 1),axis=1)

        # prediction
        x = self.fcn(x)
        x = x.view(x.shape[0], -1)

        return x #(bacth_size, num_class)

    def extract_feature(self, x):

        # B, N, M, T, V, C to N, C, T, V, M
        B, N, M, T, V, C = x.shape
        x = x.view(B*N, M, T, V, C)
        x = x.transpose(0,4,2,3,1)

        # data normalization
        N, C, T, V, M = x.shape
        x = x.permute(0, 4, 3, 1, 2)#.contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2)#.contiguous()
        x = x.view(N * M, C, T, V)

        # forwad
        for gcn in zip(self.st_gcn_networks):
            x = gcn(x, self.A)

        _, c, t, v = x.shape
        feature = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)

        # prediction
        x = self.fcn(x)
        output = x.view(N, M, -1, t, v).permute(0, 2, 3, 4, 1)

        return output, feature

class st_gcn(nn.Cell):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=1., #dropout率与pytorch定义正好相反
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, (kernel_size[0] - 1) // 2, 0, 0)
        #padding 是一个有4个整数的tuple，那么上、下、左、右的填充分别等于 padding[0] 、 padding[1] 、 padding[2] 和 padding[3]
        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.SequentialCell(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                pad_mode= "pad", # 填充模式
                padding=padding,
                has_bias=True,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(keep_prob=dropout),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.SequentialCell(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1),
                    has_bias=True),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU()

    def construct(self, x, A):

        res = self.residual(x) # (6, 1, 100, 17)

        x, A = self.gcn(x, A) # (6, 1, 100, 17) (1, 17, 17)

        x = self.tcn(x) + res

        return self.relu(x), A

if __name__=="__main__":
    # model测试
    model = STGCN(3, 500, 60, dict(layout='coco', mode='stgcn_spatial'), True)
    for para in model.parameters_dict():
        print(para)
    shape = (1, 1, 1, 500, 17, 3) # dataloader直接读取的格式
    uniformreal = mindspore.ops.UniformReal(seed=2)
    x = uniformreal(shape)
    y = model(x)
    print(y.shape)
    # (2, 1, 1, 500, 17, 3)->(2, 60)     (2, 10, 1, 500, 17, 3)->(20, 60)

    # # stgcn测试
    # st_gcn = st_gcn(3, 64, (9, 1), 1)
    # #  整个网络的输入是一个(N = batch_size,C = 3,T = 300,V = 18,M = 2)的tensor。
    # #  设 N*M(2*2)/C(3)/T(150)/V(18)
    # shape = (4, 3, 150, 18)
    # uniformreal = mindspore.ops.UniformReal(seed=2)
    # x = uniformreal(shape)
    # A = numpy.random.rand(1, 18, 18)#Graph()
    # A = Parameter(Tensor(A, dtype=mindspore.float32), requires_grad=False)
    # x, A = st_gcn(x, A)
    # print(x.shape)