import re
import timm
import torch
import torch.nn as nn

from timm.models.layers import BatchNormAct2d


class Identity(nn.Module): 

    def forward(self, x): return x


class BatchNormAct3d(nn.BatchNorm3d):
    """BatchNorm + Activation
    This module performs BatchNorm + Activation in a manner that will remain backwards
    compatible with weights trained with separate bn, act. This is why we inherit from BN
    instead of composing it as a .bn member.
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True,
                 apply_act=True, act_layer=nn.ReLU, inplace=True, drop_block=None):
        super(BatchNormAct3d, self).__init__(
            num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
        if isinstance(act_layer, str):
            act_layer = get_act_layer(act_layer)
        if act_layer is not None and apply_act:
            act_args = dict(inplace=True) if inplace else {}
            self.act = act_layer(**act_args)
        else:
            self.act = nn.Identity()
    def _forward_jit(self, x):
        """ A cut & paste of the contents of the PyTorch BatchNorm2d forward function
        """
        # exponential_average_factor is self.momentum set to
        # (when it is available) only so that if gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum
        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum
        x = F.batch_norm(
                x, self.running_mean, self.running_var, self.weight, self.bias,
                self.training or not self.track_running_stats,
                exponential_average_factor, self.eps)
        return x
    @torch.jit.ignore
    def _forward_python(self, x):
        return super(BatchNormAct3d, self).forward(x)
    def forward(self, x):
        # FIXME cannot call parent forward() and maintain jit.script compatibility?
        if torch.jit.is_scripting():
            x = self._forward_jit(x)
        else:
            x = self._forward_python(x)
        x = self.act(x)
        return x


def get_attribute(model, name):
    for i, n in enumerate(name[:-1]):
        if i == 0: 
            if isinstance(n, int):
                attr = model[n]
            else:
                attr = getattr(model, n)
        else:
            if isinstance(n, int):
                attr = attr[n]
            else:
                attr = getattr(attr, n)
    return attr


def convert_model_to_3d(model, no_stride):
    for name, module in model.named_modules():
        name = name.split('.')
        if isinstance(module, nn.Conv2d):
            w, b = module.weight, module.bias
            in_channels = module.in_channels
            out_channels = module.out_channels
            kernel_size = module.kernel_size
            w = w.unsqueeze(2).repeat(1, 1, kernel_size[0], 1, 1) / kernel_size[0]
            stride = module.stride 
            dilation = module.dilation
            padding = module.padding
            groups = module.groups
            z_stride = 1 if no_stride else stride[0]
            new_module = nn.Conv3d(in_channels,
                                   out_channels,
                                   kernel_size=(kernel_size[0], kernel_size[0], kernel_size[1]),
                                   stride=(z_stride, stride[0], stride[1]),
                                   padding=(padding[0], padding[0], padding[1]),
                                   dilation=(dilation[0], dilation[0], dilation[1]),
                                   bias=isinstance(b, torch.Tensor),
                                   groups=groups)
            assert w.shape == new_module.weight.shape, f'Inflated weight shape {w.shape} does not match module weight shape {new_module.weight.shape}'
            _ = new_module.weight.data.copy_(w)
            if isinstance(b, torch.Tensor):
                _ = new_module.bias.data.copy_(b)
        if isinstance(module, nn.BatchNorm2d):
            w, b = module.weight, module.bias
            m, v = module.running_mean, module.running_var
            num_features = module.num_features
            eps = module.eps
            momentum = module.momentum
            affine = module.affine
            track_running_stats = module.track_running_stats
            if isinstance(module, BatchNormAct2d):
                new_module = BatchNormAct3d(num_features, eps, momentum, affine, track_running_stats, act_layer=type(module.act))
            else:
                new_module = nn.BatchNorm3d(num_features, eps, momentum, affine, track_running_stats)
            _ = new_module.weight.data.copy_(w)
            _ = new_module.bias.data.copy_(b)
            _ = new_module.running_mean.data.copy_(m)
            _ = new_module.running_var.data.copy_(v)
        if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
            if len(name) > 1:
                module_to_change = get_attribute(model, name) if len(name) > 1 else module
                setattr(module_to_change, name[-1], new_module)
            else:
                setattr(model, name[0], new_module)
    if hasattr(model, 'global_pool'):
        model.global_pool = nn.Identity()
    elif hasattr(model, 'head'):
        assert hasattr(model.head, 'global_pool')
        model.head.global_pool = nn.Identity()
    else:
        raise Exception('Model does not have `global_pool` or `head.global_pool` module')
    return model


class AveragePool3d(nn.Module):

    def forward(self, x):
        return torch.nn.functional.adaptive_avg_pool3d(x, 1).view(x.size(0), x.size(1), -1)


def get_backbone(name, pretrained, **kwargs):
    if name in ['i3d', 'mc3_18', 'r2plus1d_18', 'r3d_18', 'r2plus1d_34']: 
        return eval(name)(pretrained, **kwargs)
    if 'ir_csn' in name or 'ip_csn' in name:
        return eval(name)(pretrained, **kwargs)

    model = timm.create_model(name, pretrained=pretrained)
    if 'regnet' in name or 'rexnet' in name:
        dim_feats = model.head.fc.in_features
        model.head.fc = Identity()
    elif 'resnest' in name:
        dim_feats = model.fc.in_features
        model.fc = Identity()
    elif "swin" in name:
        dim_feats = model.head.in_features 
        model.head = Identity()
    else:
        dim_feats = model.classifier.in_features
        model.classifier = Identity()
    return model, dim_feats


#############
# 3D MODELS #
#############
from torchvision.models.video import mc3_18 as _mc3_18
from torchvision.models.video import r2plus1d_18 as _r2plus1d_18
from torchvision.models.video import r3d_18 as _r3d_18

from . import vmz
from .i3d import InceptionI3d


def i3d(pretrained=True, **kwargs):
    model = InceptionI3d()
    if pretrained:
        weights = torch.hub.load_state_dict_from_url('https://github.com/piergiaj/pytorch-i3d/raw/master/models/rgb_imagenet.pt')
        model.load_state_dict(weights)
        model.avg_pool = nn.AdaptiveAvgPool3d(1)
    model.logits = Identity()
    return model, 1024


def mc3_18(pretrained=True, **kwargs):
    model = _mc3_18(pretrained=pretrained)
    dim_feats = model.fc.in_features
    model.fc = Identity()
    return model, dim_feats


def r2plus1d_18(pretrained=True, **kwargs):
    model = _r2plus1d_18(pretrained=pretrained)
    dim_feats = model.fc.in_features
    model.fc = Identity()
    return model, dim_feats


def r3d_18(pretrained=True, **kwargs):
    model = _r3d_18(pretrained=pretrained)
    dim_feats = model.fc.in_features
    model.fc = Identity()
    return model, dim_feats


def ir_csn_152(pretrained=True, **kwargs):
    model = vmz.ir_csn_152(pretraining='ig65m_32frms' if pretrained else '', num_classes=359)
    dim_feats = model.fc.in_features
    model.fc = Identity() 
    return model, dim_feats


def ir_csn_101(pretrained=True, **kwargs):
    model = vmz.ir_csn_152(pretraining='ig65m_32frms' if pretrained else '', num_classes=359)
    dim_feats = model.fc.in_features
    model.fc = Identity()
    model.layer2 = model.layer2[:4]
    model.layer3 = model.layer3[:23]
    return model, dim_feats


def ir_csn_50(pretrained=True, **kwargs):
    model = vmz.ir_csn_152(pretraining='ig65m_32frms' if pretrained else '', num_classes=359)
    dim_feats = model.fc.in_features
    model.fc = Identity()
    model.layer2 = model.layer2[:4]
    model.layer3 = model.layer3[:6]
    return model, dim_feats


def ip_csn_152(pretrained=True, **kwargs):
    model = vmz.ip_csn_152(pretraining='ig65m_32frms' if pretrained else '', num_classes=359)
    dim_feats = model.fc.in_features
    model.fc = Identity() 
    return model, dim_feats


def ip_csn_101(pretrained=True, **kwargs):
    model = vmz.ip_csn_152(pretraining='ig65m_32frms' if pretrained else '', num_classes=359)
    dim_feats = model.fc.in_features
    model.fc = Identity()
    model.layer2 = model.layer2[:4]
    model.layer3 = model.layer3[:23]
    return model, dim_feats


def ip_csn_50(pretrained=True, **kwargs):
    model = vmz.ip_csn_152(pretraining='ig65m_32frms' if pretrained else '', num_classes=359)
    dim_feats = model.fc.in_features
    model.fc = Identity()
    model.layer2 = model.layer2[:4]
    model.layer3 = model.layer3[:6]
    return model, dim_feats


def r2plus1d_34(pretrained=True, **kwargs):
    model = vmz.r2plus1d_34(pretraining='32_ig65m' if pretrained else '', num_classes=359)
    dim_feats = model.fc.in_features
    model.fc = Identity() 
    return model, dim_feats