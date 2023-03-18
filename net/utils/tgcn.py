# The based unit of graph convolutional networks.

import mindspore
import mindspore.nn as nn
from mindspore import Parameter, Tensor
import numpy.random


class ConvTemporalGraphical(nn.Cell):

    r"""The basic module for applying a graph convolution.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``

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
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0, 0, 0),#原本size是(0, 0),按理来说多个0,0padding没事
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            has_bias=bias)

    def construct(self, x, A):
        assert A.shape[0] == self.kernel_size

        #  这里输入x是(N,C,T,V),经过conv(x)之后变为（N，C*kneral_size,T,V）
        x = self.conv(x)

        n, kc, t, v = x.shape
        x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)

        #x = torch.einsum('nkctv,kvw->nctw', (x, A)) 不支持，之后修正

        return x, A

if __name__=="__main__":
    gcn = ConvTemporalGraphical(3, 64, 1)
    #  设 N=1, C=3, T=300, V=18
    shape = (512, 3, 150, 18)
    uniformreal = mindspore.ops.UniformReal(seed=2)
    x = uniformreal(shape)
    A = numpy.random.rand(1, 18, 18)#Graph()
    A = Parameter(Tensor(A, dtype=mindspore.float32), requires_grad=False)
    x, A = gcn(x, A)
    print(x.shape, A.shape)


