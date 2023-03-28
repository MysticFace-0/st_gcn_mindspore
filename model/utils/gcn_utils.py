# Copyright (c) OpenMMLab. All rights reserved.
import copy as cp
from typing import Dict, List, Optional, Union

import mindspore
import torch
import torch.nn as nn

def build_norm_layer(cfg: Dict, # {'type': 'BN'}
                     num_features: int, # 64
                     postfix: Union[int, str] = '') -> tuple[str, nn.Module]:
    """Build normalization layer.

    Args:
        cfg (dict): The norm layer config, which should contain:

            - type (str): Layer type.
            - layer args: Args needed to instantiate a norm layer.
            - requires_grad (bool, optional): Whether stop gradient updates.
        num_features (int): Number of input channels.
        postfix (int | str): The postfix to be appended into norm abbreviation
            to create named layer.

    Returns:
        tuple[str, nn.Module]: The first element is the layer name consisting
        of abbreviation and postfix, e.g., bn1, gn. The second element is the
        created norm layer.
    """
    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in NORM_LAYERS:
        raise KeyError(f'Unrecognized norm type {layer_type}')

    norm_layer = NORM_LAYERS.get(layer_type)
    abbr = infer_abbr(norm_layer)

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    requires_grad = cfg_.pop('requires_grad', True)
    cfg_.setdefault('eps', 1e-5)
    if layer_type != 'GN':
        layer = norm_layer(num_features, **cfg_)
        if layer_type == 'SyncBN' and hasattr(layer, '_specify_ddp_gpu_num'):
            layer._specify_ddp_gpu_num(1)
    else:
        assert 'num_groups' in cfg_
        layer = norm_layer(num_channels=num_features, **cfg_)

    for param in layer.parameters():
        param.requires_grad = requires_grad

    return name, layer

def build_from_cfg(cfg: Dict,
                   registry: 'Registry',
                   default_args: Optional[Dict] = None) -> Any:
    """Build a module from config dict when it is a class configuration, or
    call a function from config dict when it is a function configuration.

    Example:
        >>> MODELS = Registry('models')
        >>> @MODELS.register_module()
        >>> class ResNet:
        >>>     pass
        >>> resnet = build_from_cfg(dict(type='Resnet'), MODELS)
        >>> # Returns an instantiated object
        >>> @MODELS.register_module()
        >>> def resnet50():
        >>>     pass
        >>> resnet = build_from_cfg(dict(type='resnet50'), MODELS)
        >>> # Return a result of the calling function

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        registry (:obj:`Registry`): The registry to search the type from.
        default_args (dict, optional): Default initialization arguments.

    Returns:
        object: The constructed object.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f'cfg must be a dict, but got {type(cfg)}')
    if 'type' not in cfg:
        if default_args is None or 'type' not in default_args:
            raise KeyError(
                '`cfg` or `default_args` must contain the key "type", '
                f'but got {cfg}\n{default_args}')
    if not isinstance(registry, Registry):
        raise TypeError('registry must be an mmcv.Registry object, '
                        f'but got {type(registry)}')
    if not (isinstance(default_args, dict) or default_args is None):
        raise TypeError('default_args must be a dict or None, '
                        f'but got {type(default_args)}')

    args = cfg.copy()

    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)

    obj_type = args.pop('type')
    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError(
                f'{obj_type} is not in the {registry.name} registry')
    elif inspect.isclass(obj_type) or inspect.isfunction(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(
            f'type must be a str or valid type, but got {type(obj_type)}')
    try:
        return obj_cls(**args)
    except Exception as e:
        # Normal TypeError does not print class name.
        raise type(e)(f'{obj_cls.__name__}: {e}')

def build_activation_layer(cfg: Dict) -> nn.Module:
    """Build activation layer.

    Args:
        cfg (dict): The activation layer config, which should contain:

            - type (str): Layer type.
            - layer args: Args needed to instantiate an activation layer.

    Returns:
        nn.Module: Created activation layer.
    """
    return build_from_cfg(cfg, ACTIVATION_LAYERS)

class unit_gcn(nn.Cell):
    """The basic unit of graph convolutional network.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        A (torch.Tensor): The adjacency matrix defined in the graph
            with shape of `(num_subsets, num_nodes, num_nodes)`.
        adaptive (str): The strategy for adapting the weights of the
            adjacency matrix. Defaults to ``'importance'``.
        conv_pos (str): The position of the 1x1 2D conv.
            Defaults to ``'pre'``.
        with_res (bool): Whether to use residual connection.
            Defaults to False.
        norm (str): The name of norm layer. Defaults to ``'BN'``.
        act (str): The name of activation layer. Defaults to ``'Relu'``.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: int, # 3
                 out_channels: int, # 64
                 A: mindspore.Tensor, # (3,17,17)
                 adaptive: str = 'importance',
                 conv_pos: str = 'pre',
                 with_res: bool = False,
                 norm: str = 'BN',
                 act: str = 'ReLU') -> None:
        super().__init__()
        self.in_channels = in_channels # 3
        self.out_channels = out_channels # 64
        self.num_subsets = A.size(0)

        assert adaptive in [None, 'init', 'offset', 'importance']
        self.adaptive = adaptive # importance
        assert conv_pos in ['pre', 'post']
        self.conv_pos = conv_pos # pre
        self.with_res = with_res # False

        self.norm_cfg = norm if isinstance(norm, dict) else dict(type=norm) # {'type': 'BN'}
        self.act_cfg = act if isinstance(act, dict) else dict(type=act) # {'type': 'RELU'}
        self.bn = build_norm_layer(self.norm_cfg, out_channels)[1]
        self.act = build_activation_layer(self.act_cfg)

        if self.adaptive == 'init':
            self.A = nn.Parameter(A.clone())
        else:
            self.register_buffer('A', A)

        if self.adaptive in ['offset', 'importance']:
            self.PA = nn.Parameter(A.clone())
            if self.adaptive == 'offset':
                nn.init.uniform_(self.PA, -1e-6, 1e-6)
            elif self.adaptive == 'importance':
                nn.init.constant_(self.PA, 1)

        if self.conv_pos == 'pre':
            self.conv = nn.Conv2d(in_channels, out_channels * A.size(0), 1)
        elif self.conv_pos == 'post':
            self.conv = nn.Conv2d(A.size(0) * in_channels, out_channels, 1)

        if self.with_res:
            if in_channels != out_channels:
                self.down = Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    build_norm_layer(self.norm_cfg, out_channels)[1])
            else:
                self.down = lambda x: x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call."""
        n, c, t, v = x.shape
        res = self.down(x) if self.with_res else 0

        A_switch = {None: self.A, 'init': self.A}
        if hasattr(self, 'PA'):
            A_switch.update({
                'offset': self.A + self.PA,
                'importance': self.A * self.PA
            })
        A = A_switch[self.adaptive]

        if self.conv_pos == 'pre':
            x = self.conv(x)
            x = x.view(n, self.num_subsets, -1, t, v)
            x = torch.einsum('nkctv,kvw->nctw', (x, A)).contiguous()
        elif self.conv_pos == 'post':
            x = torch.einsum('nctv,kvw->nkctw', (x, A)).contiguous()
            x = x.view(n, -1, t, v)
            x = self.conv(x)

        return self.act(self.bn(x) + res)


class unit_tcn(nn.Cell):
    """The basic unit of temporal convolutional network.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the temporal convolution kernel.
            Defaults to 9.
        stride (int): Stride of the temporal convolution. Defaults to 1.
        dilation (int): Spacing between temporal kernel elements.
            Defaults to 1.
        norm (str): The name of norm layer. Defaults to ``'BN'``.
        dropout (float): Dropout probability. Defaults to 0.
        init_cfg (dict or list[dict]): Initialization config dict. Defaults to
            ``[
                dict(type='Constant', layer='BatchNorm2d', val=1),
                dict(type='Kaiming', layer='Conv2d', mode='fan_out')
            ]``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 9,
        stride: int = 1,
        dilation: int = 1,
        norm: str = 'BN',
        dropout: float = 0,
        init_cfg: Union[Dict, List[Dict]] = [
            dict(type='Constant', layer='BatchNorm2d', val=1),
            dict(type='Kaiming', layer='Conv2d', mode='fan_out')
        ]
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_cfg = norm if isinstance(norm, dict) else dict(type=norm)
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))
        self.bn = build_norm_layer(self.norm_cfg, out_channels)[1] \
            if norm is not None else nn.Identity()

        self.drop = nn.Dropout(dropout, inplace=True)
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call."""
        return self.drop(self.bn(self.conv(x)))


class mstcn(nn.Cell):
    """The multi-scale temporal convolutional network.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        mid_channels (int): Number of middle channels. Defaults to None.
        dropout (float): Dropout probability. Defaults to 0.
        ms_cfg (list): The config of multi-scale branches. Defaults to
            ``[(3, 1), (3, 2), (3, 3), (3, 4), ('max', 3), '1x1']``.
        stride (int): Stride of the temporal convolution. Defaults to 1.
        init_cfg (dict or list[dict]): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 mid_channels: int = None,
                 dropout: float = 0.,
                 ms_cfg: List = [(3, 1), (3, 2), (3, 3), (3, 4), ('max', 3),
                                 '1x1'],
                 stride: int = 1,
                 init_cfg: Union[Dict, List[Dict]] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        # Multiple branches of temporal convolution
        self.ms_cfg = ms_cfg
        num_branches = len(ms_cfg)
        self.num_branches = num_branches
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = nn.ReLU()

        if mid_channels is None:
            mid_channels = out_channels // num_branches
            rem_mid_channels = out_channels - mid_channels * (num_branches - 1)
        else:
            assert isinstance(mid_channels, float) and mid_channels > 0
            mid_channels = int(out_channels * mid_channels)
            rem_mid_channels = mid_channels

        self.mid_channels = mid_channels
        self.rem_mid_channels = rem_mid_channels

        branches = []
        for i, cfg in enumerate(ms_cfg):
            branch_c = rem_mid_channels if i == 0 else mid_channels
            if cfg == '1x1':
                branches.append(
                    nn.Conv2d(
                        in_channels,
                        branch_c,
                        kernel_size=1,
                        stride=(stride, 1)))
                continue
            assert isinstance(cfg, tuple)
            if cfg[0] == 'max':
                branches.append(
                    Sequential(
                        nn.Conv2d(in_channels, branch_c, kernel_size=1),
                        nn.BatchNorm2d(branch_c), self.act,
                        nn.MaxPool2d(
                            kernel_size=(cfg[1], 1),
                            stride=(stride, 1),
                            padding=(1, 0))))
                continue
            assert isinstance(cfg[0], int) and isinstance(cfg[1], int)
            branch = Sequential(
                nn.Conv2d(in_channels, branch_c, kernel_size=1),
                nn.BatchNorm2d(branch_c), self.act,
                unit_tcn(
                    branch_c,
                    branch_c,
                    kernel_size=cfg[0],
                    stride=stride,
                    dilation=cfg[1],
                    norm=None))
            branches.append(branch)

        self.branches = ModuleList(branches)
        tin_channels = mid_channels * (num_branches - 1) + rem_mid_channels

        self.transform = Sequential(
            nn.BatchNorm2d(tin_channels), self.act,
            nn.Conv2d(tin_channels, out_channels, kernel_size=1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout(dropout, inplace=True)

    def inner_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call."""
        N, C, T, V = x.shape

        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        feat = torch.cat(branch_outs, dim=1)
        feat = self.transform(feat)
        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call."""
        out = self.inner_forward(x)
        out = self.bn(out)
        return self.drop(out)
