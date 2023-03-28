# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import mindspore
import torch
from mindspore import ops

from model.backbones import STGCN


class RecognizerGCN():
    """GCN-based recognizer for skeleton-based action recognition."""

    def __init__(self):
        self.backbone = STGCN(graph_cfg=dict(layout='coco', mode='spatial'))
        self.cls_head = None

    def with_cls_head(self) -> bool:
        """bool: whether the recognizer has a cls_head"""
        return hasattr(self, 'cls_head') and self.cls_head is not None

    def extract_feat(self,
                     inputs: mindspore.Tensor,
                     stage: str = 'backbone') -> Tuple:
        """Extract features at the given stage.

        Args:
            inputs (torch.Tensor): The input skeleton with shape of
                `(B, num_clips, num_person, clip_len, num_joints, 3 or 2)`.
            stage (str): The stage to output the features.
                Defaults to ``'backbone'``.

        Returns:
            tuple: THe extracted features and a dict recording the kwargs
            for downstream pipeline, which is an empty dict for the
            GCN-based recognizer.
        """

        bs, nc = inputs.shape[:2] # batch_size, num_clips
        inputs = inputs.reshape((bs * nc, ) + inputs.shape[2:]) # (2*10, 1, 500, 17, 3)

        x = self.backbone(inputs) # STGCN

        if stage == 'backbone':
            return x

        if self.with_cls_head and stage == 'head': # not finished
            x = self.cls_head(x)
            return x

if __name__=="__main__":
    shape = (2, 5, 1, 500, 17, 3)
    uniformreal = ops.UniformReal(seed=2)
    inputs = uniformreal(shape)
    RecognizerGCN = RecognizerGCN()
    y = RecognizerGCN.extract_feat(inputs)
    print(y.shape)

