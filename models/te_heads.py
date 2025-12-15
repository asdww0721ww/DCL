# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn

from abc import ABCMeta, abstractmethod

import torch.nn.functional as F
# from mmcv.runner import force_fp32

# from mmcv.cnn import ConvModule
# from mmcv.runner import BaseModule, ModuleList
from .te_neck import BaseModule, ConvModule
from torch.nn import ModuleList

def force_fp32(apply_to=None, out_fp16=False):
    def check_fp32(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return check_fp32


# Copyright (c) OpenMMLab. All rights reserved.
from torch.autograd import Function
from torch.nn import functional as F



def interpolate_as(source, target, mode='bilinear', align_corners=False):
    """Interpolate the `source` to the shape of the `target`.

    The `source` must be a Tensor, but the `target` can be a Tensor or a
    np.ndarray with the shape (..., target_h, target_w).

    Args:
        source (Tensor): A 3D/4D Tensor with the shape (N, H, W) or
            (N, C, H, W).
        target (Tensor | np.ndarray): The interpolation target with the shape
            (..., target_h, target_w).
        mode (str): Algorithm used for interpolation. The options are the
            same as those in F.interpolate(). Default: ``'bilinear'``.
        align_corners (bool): The same as the argument in F.interpolate().

    Returns:
        Tensor: The interpolated source Tensor.
    """
    assert len(target.shape) >= 2

    def _interpolate_as(source, target, mode='bilinear', align_corners=False):
        """Interpolate the `source` (4D) to the shape of the `target`."""
        target_h, target_w = target.shape[-2:]
        source_h, source_w = source.shape[-2:]
        if target_h != source_h or target_w != source_w:
            source = F.interpolate(
                source,
                size=(target_h, target_w),
                mode=mode,
                align_corners=align_corners)
        return source

    if len(source.shape) == 3:
        source = source[:, None, :, :]
        source = _interpolate_as(source, target, mode, align_corners)
        return source[:, 0, :, :]
    else:
        return _interpolate_as(source, target, mode, align_corners)

class ConvUpsample(BaseModule):
    """ConvUpsample performs 2x upsampling after Conv.

    There are several `ConvModule` layers. In the first few layers, upsampling
    will be applied after each layer of convolution. The number of upsampling
    must be no more than the number of ConvModule layers.

    Args:
        in_channels (int): Number of channels in the input feature map.
        inner_channels (int): Number of channels produced by the convolution.
        num_layers (int): Number of convolution layers.
        num_upsample (int | optional): Number of upsampling layer. Must be no
            more than num_layers. Upsampling will be applied after the first
            ``num_upsample`` layers of convolution. Default: ``num_layers``.
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        init_cfg (dict): Config dict for initialization. Default: None.
        kwargs (key word augments): Other augments used in ConvModule.
    """

    def __init__(self,
                 in_channels,
                 inner_channels,
                 num_layers=1,
                 num_upsample=None,
                 conv_cfg=None,
                 norm_cfg=None,
                 init_cfg=None,
                 **kwargs):
        super(ConvUpsample, self).__init__(init_cfg)
        if num_upsample is None:
            num_upsample = num_layers
        assert num_upsample <= num_layers, \
            f'num_upsample({num_upsample})must be no more than ' \
            f'num_layers({num_layers})'
        self.num_layers = num_layers
        self.num_upsample = num_upsample
        self.conv = ModuleList()
        for i in range(num_layers):
            self.conv.append(
                ConvModule(
                    in_channels,
                    inner_channels,
                    3,
                    padding=1,
                    stride=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    **kwargs))
            in_channels = inner_channels

    def forward(self, x):
        num_upsample = self.num_upsample
        for i in range(self.num_layers):
            x = self.conv[i](x)
            if num_upsample > 0:
                num_upsample -= 1
                x = F.interpolate(
                    x, scale_factor=2, mode='bilinear', align_corners=False)
        return x

class BaseSemanticHead(BaseModule, metaclass=ABCMeta):
    """Base module of Semantic Head.

    Args:
        num_classes (int): the number of classes.
        init_cfg (dict): the initialization config.
        loss_seg (dict): the loss of the semantic head.
    """

    def __init__(self,
                 num_classes,
                 init_cfg=None,
                 loss_seg=dict(
                     type='CrossEntropyLoss',
                     ignore_index=255,
                     loss_weight=1.0)):
        super(BaseSemanticHead, self).__init__(init_cfg)
        # self.loss_seg = build_loss(loss_seg)
        self.num_classes = num_classes

    @force_fp32(apply_to=('seg_preds', ))
    def loss(self, seg_preds, gt_semantic_seg):
        """Get the loss of semantic head.

        Args:
            seg_preds (Tensor): The input logits with the shape (N, C, H, W).
            gt_semantic_seg: The ground truth of semantic segmentation with
                the shape (N, H, W).
            label_bias: The starting number of the semantic label.
                Default: 1.

        Returns:
            dict: the loss of semantic head.
        """
        if seg_preds.shape[-2:] != gt_semantic_seg.shape[-2:]:
            seg_preds = interpolate_as(seg_preds, gt_semantic_seg)
        seg_preds = seg_preds.permute((0, 2, 3, 1))

        loss_seg = self.loss_seg(
            seg_preds.reshape(-1, self.num_classes),  # => [NxHxW, C]
            gt_semantic_seg.reshape(-1).long())
        return dict(loss_seg=loss_seg)

    @abstractmethod
    def forward(self, x):
        """Placeholder of forward function.

        Returns:
            dict[str, Tensor]: A dictionary, including features
                and predicted scores. Required keys: 'seg_preds'
                and 'feats'.
        """
        pass

    def forward_train(self, x, gt_semantic_seg):
        output = self.forward(x)
        seg_preds = output['seg_preds']
        return self.loss(seg_preds, gt_semantic_seg)

    def simple_test(self, x, img_metas, rescale=False):
        output = self.forward(x)
        seg_preds = output['seg_preds']
        seg_preds = F.interpolate(
            seg_preds,
            size=img_metas[0]['pad_shape'][:2],
            mode='bilinear',
            align_corners=False)

        if rescale:
            h, w, _ = img_metas[0]['img_shape']
            seg_preds = seg_preds[:, :, :h, :w]

            h, w, _ = img_metas[0]['ori_shape']
            seg_preds = F.interpolate(
                seg_preds, size=(h, w), mode='bilinear', align_corners=False)
        return seg_preds


class PanopticFPNHead(BaseSemanticHead):
    """PanopticFPNHead used in Panoptic FPN.

    In this head, the number of output channels is ``num_stuff_classes
    + 1``, including all stuff classes and one thing class. The stuff
    classes will be reset from ``0`` to ``num_stuff_classes - 1``, the
    thing classes will be merged to ``num_stuff_classes``-th channel.

    Arg:
        num_things_classes (int): Number of thing classes. Default: 80.
        num_stuff_classes (int): Number of stuff classes. Default: 53.
        num_classes (int): Number of classes, including all stuff
            classes and one thing class. This argument is deprecated,
            please use ``num_things_classes`` and ``num_stuff_classes``.
            The module will automatically infer the num_classes by
            ``num_stuff_classes + 1``.
        in_channels (int): Number of channels in the input feature
            map.
        inner_channels (int): Number of channels in inner features.
        start_level (int): The start level of the input features
            used in PanopticFPN.
        end_level (int): The end level of the used features, the
            ``end_level``-th layer will not be used.
        fg_range (tuple): Range of the foreground classes. It starts
            from ``0`` to ``num_things_classes-1``. Deprecated, please use
             ``num_things_classes`` directly.
        bg_range (tuple): Range of the background classes. It starts
            from ``num_things_classes`` to ``num_things_classes +
            num_stuff_classes - 1``. Deprecated, please use
            ``num_stuff_classes`` and ``num_things_classes`` directly.
        conv_cfg (dict): Dictionary to construct and config
            conv layer. Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Use ``GN`` by default.
        init_cfg (dict or list[dict], optional): Initialization config dict.
        loss_seg (dict): the loss of the semantic head.
    """

    def __init__(self,
                 num_things_classes=80,
                 num_stuff_classes=53,
                 num_classes=None,
                 in_channels=256,
                 inner_channels=128,
                 start_level=0,
                 end_level=4,
                 fg_range=None,
                 bg_range=None,
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 init_cfg=None,
                 loss_seg=dict(
                     type='CrossEntropyLoss', ignore_index=-1,
                     loss_weight=1.0)):
        if num_classes is not None:
            warnings.warn(
                '`num_classes` is deprecated now, please set '
                '`num_stuff_classes` directly, the `num_classes` will be '
                'set to `num_stuff_classes + 1`')
            # num_classes = num_stuff_classes + 1 for PanopticFPN.
            assert num_classes == num_stuff_classes + 1
        super(PanopticFPNHead, self).__init__(num_stuff_classes + 1, init_cfg,
                                              loss_seg)
        self.num_things_classes = num_things_classes
        self.num_stuff_classes = num_stuff_classes
        if fg_range is not None and bg_range is not None:
            self.fg_range = fg_range
            self.bg_range = bg_range
            self.num_things_classes = fg_range[1] - fg_range[0] + 1
            self.num_stuff_classes = bg_range[1] - bg_range[0] + 1
            warnings.warn(
                '`fg_range` and `bg_range` are deprecated now, '
                f'please use `num_things_classes`={self.num_things_classes} '
                f'and `num_stuff_classes`={self.num_stuff_classes} instead.')

        # Used feature layers are [start_level, end_level)
        self.start_level = start_level
        self.end_level = end_level
        self.num_stages = end_level - start_level
        self.inner_channels = inner_channels

        self.conv_upsample_layers = ModuleList()
        for i in range(start_level, end_level):
            self.conv_upsample_layers.append(
                ConvUpsample(
                    in_channels,
                    inner_channels,
                    num_layers=i if i > 0 else 1,
                    num_upsample=i if i > 0 else 0,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                ))
        self.conv_logits = nn.Conv2d(inner_channels, self.num_classes, 1)

    def _set_things_to_void(self, gt_semantic_seg):
        """Merge thing classes to one class.

        In PanopticFPN, the background labels will be reset from `0` to
        `self.num_stuff_classes-1`, the foreground labels will be merged to
        `self.num_stuff_classes`-th channel.
        """
        gt_semantic_seg = gt_semantic_seg.int()
        fg_mask = gt_semantic_seg < self.num_things_classes
        bg_mask = (gt_semantic_seg >= self.num_things_classes) * (
            gt_semantic_seg < self.num_things_classes + self.num_stuff_classes)

        new_gt_seg = torch.clone(gt_semantic_seg)
        new_gt_seg = torch.where(bg_mask,
                                 gt_semantic_seg - self.num_things_classes,
                                 new_gt_seg)
        new_gt_seg = torch.where(fg_mask,
                                 fg_mask.int() * self.num_stuff_classes,
                                 new_gt_seg)
        return new_gt_seg

    def init_weights(self):
        super().init_weights()
        nn.init.normal_(self.conv_logits.weight.data, 0, 0.01)
        self.conv_logits.bias.data.zero_()

    def forward(self, x):
        # the number of subnets must be not more than
        # the length of features.
        assert self.num_stages <= len(x)

        feats = []
        for i, layer in enumerate(self.conv_upsample_layers):
            f = layer(x[self.start_level + i])
            feats.append(f)

        feats = torch.sum(torch.stack(feats, dim=0), dim=0)
        seg_preds = self.conv_logits(feats)
        out = dict(seg_preds=seg_preds, feats=feats)
        return out