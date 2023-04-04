import mindspore
import mindspore.nn as nn
from mindspore.common.initializer import initializer, HeNormal
import torch

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

tcn1 = unit_tcn(3, 64, stride=1)
shape = (2, 3, 100, 17)
uniformreal = mindspore.ops.UniformReal(seed=2)
x = uniformreal(shape)
y = tcn1(x)
print(y.shape)

