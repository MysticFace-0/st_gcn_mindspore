# Copyright (c) OpenMMLab. All rights reserved.
import copy as cp
from typing import Dict, List, Optional, Union

import mindspore
import mindspore.nn as nn
import numpy
from mindspore import Parameter, Tensor, ops

# from ..utils import Graph, mstcn, unit_gcn, unit_tcn

EPS = 1e-4

class STGCNBlock(nn.Cell):
    """The basic block of STGCN.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        A (torch.Tensor): The adjacency matrix defined in the graph
            with shape of `(num_subsets, num_nodes, num_nodes)`.
        stride (int): Stride of the temporal convolution. Defaults to 1.
        residual (bool): Whether to use residual connection. Defaults to True.
        init_cfg (dict or list[dict], optional): Config to control
            the initialization. Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 A: mindspore.Tensor,
                 stride: int = 1,
                 residual: bool = True,
                 **kwargs) -> None:
        super().__init__()

        gcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'gcn_'}
        tcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'tcn_'}
        kwargs = {
            k: v
            for k, v in kwargs.items() if k[:4] not in ['gcn_', 'tcn_']
        }
        assert len(kwargs) == 0, f'Invalid arguments: {kwargs}'

        tcn_type = tcn_kwargs.pop('type', 'unit_tcn')
        assert tcn_type in ['unit_tcn', 'mstcn']
        gcn_type = gcn_kwargs.pop('type', 'unit_gcn')
        assert gcn_type in ['unit_gcn']

        self.gcn = unit_gcn(in_channels, out_channels, A, **gcn_kwargs)

        if tcn_type == 'unit_tcn':
            self.tcn = unit_tcn(
                out_channels, out_channels, 9, stride=stride, **tcn_kwargs)
        elif tcn_type == 'mstcn':
            self.tcn = mstcn(
                out_channels, out_channels, stride=stride, **tcn_kwargs)
        self.relu = nn.ReLU()

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(
                in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x: mindspore.Tensor) -> mindspore.Tensor:
        """Defines the computation performed at every call."""
        res = self.residual(x)
        x = self.tcn(self.gcn(x)) + res
        return self.relu(x)


class STGCN(nn.Cell):
    """STGCN backbone.

    Spatial Temporal Graph Convolutional
    Networks for Skeleton-Based Action Recognition.
    More details can be found in the `paper
    <https://arxiv.org/abs/1801.07455>`__ .

    Args:
        graph_cfg (dict): Config for building the graph.
        in_channels (int): Number of input channels. Defaults to 3.
        base_channels (int): Number of base channels. Defaults to 64.
        data_bn_type (str): Type of the data bn layer. Defaults to ``'VC'``.
        ch_ratio (int): Inflation ratio of the number of channels.
            Defaults to 2.
        num_person (int): Maximum number of people. Only used when
            data_bn_type == 'MVC'. Defaults to 2.
        num_stages (int): Total number of stages. Defaults to 10.
        inflate_stages (list[int]): Stages to inflate the number of channels.
            Defaults to ``[5, 8]``.
        down_stages (list[int]): Stages to perform downsampling in
            the time dimension. Defaults to ``[5, 8]``.
        stage_cfgs (dict): Extra config dict for each stage.
            Defaults to ``dict()``.
        init_cfg (dict or list[dict], optional): Config to control
            the initialization. Defaults to None.

        Examples:
        >>> # coco layout
        >>> num_joints = 17
        >>> model = STGCN(graph_cfg=dict(layout='coco', mode=mode))
        >>> model.init_weights()
        >>> inputs = (batch_size, num_person, num_frames, num_joints, 3)
        >>> output = model(inputs)
        >>> print(output.shape)
    """

    def __init__(self,
                 graph_cfg: Dict,
                 in_channels: int = 3,
                 base_channels: int = 64,
                 data_bn_type: str = 'VC',
                 ch_ratio: int = 2,
                 num_person: int = 1,
                 num_stages: int = 10,
                 inflate_stages: List[int] = [5, 8],
                 down_stages: List[int] = [5, 8],
                 **kwargs) -> None:
        super().__init__()

        self.graph = numpy.random.random([3,17,17])# Graph(**graph_cfg) # build stgcn g

        A = Parameter(Tensor(self.graph, dtype=mindspore.float32), requires_grad=False) # (3, 17, 17)

        self.data_bn_type = data_bn_type # 'VC'

        if data_bn_type == 'MVC':
            self.data_bn = nn.BatchNorm1d(num_person * in_channels * A.shape[1])
        elif data_bn_type == 'VC':
            self.data_bn = nn.BatchNorm1d(in_channels * A.shape[1])
            # 接口输入，功能基本一致，但PyTorch里允许输入是二维或三维的，而MindSpore里的输入只能是二维的
        else:
            self.data_bn = nn.Identity()

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.ch_ratio = ch_ratio
        self.inflate_stages = inflate_stages
        self.down_stages = down_stages

        modules = []
        if self.in_channels != self.base_channels:
            modules = [
                # STGCNBlock(
                #     in_channels,
                #     base_channels,
                #     A, # A.clone() 测试过mindspore中不用clone的效果是一样的
                #     1,
                #     residual=False)
            ]

        inflate_times = 0
        for i in range(2, num_stages + 1):
            stride = 1 + (i in down_stages)
            in_channels = base_channels
            if i in inflate_stages:
                inflate_times += 1
            out_channels = int(self.base_channels *
                               self.ch_ratio**inflate_times + EPS)
            base_channels = out_channels
            A_clone = A
            # modules.append(
            #     STGCNBlock(in_channels, out_channels, A_clone, stride))

        if self.in_channels == self.base_channels:
            num_stages -= 1

        self.num_stages = num_stages
        self.gcn = nn.CellList(modules)

    def construct(self, x: mindspore.Tensor) -> mindspore.Tensor:
        """Defines the computation performed at every call."""
        N, M, T, V, C = x.shape
        x = x.permute(0, 1, 3, 4, 2)# .contiguous() # (2, 1, 17, 3, 100)
        if self.data_bn_type == 'MVC':
            x = self.data_bn(x.view(N, M * V * C, T))
        else:
            x = self.data_bn(x.view(N * M, V * C, T))

        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2) #.contiguous()
        x = x.view(N * M, C, T, V)

        for i in range(self.num_stages):
            x = self.gcn[i](x)

        reshape = ops.Reshape()  # 变形算子
        x = reshape(x, (N, M) + x.shape[1:])
        return x

if __name__=="__main__":
    num_joints = 17
    model = STGCN(num_person=1, graph_cfg=dict(layout='coco', mode='stgcn_spacial'))
    shape = (2*1, 1, 100, 17, 3)
    uniformreal = ops.UniformReal(seed=2)
    inputs = uniformreal(shape)
    output = model(inputs)
    print(output.shape)