import re
import torch
import torch.nn as nn
import torch.nn.functional as F

#from timesformer.models.vit import TimeSformer as _TimeSformer

from . import backbones
#from . import smp
from .constants import POOLING
from .pooling import GeM, AdaptiveConcatPool2d, AdaptiveAvgPool2d, AdaptiveAvgPool25d, AdaptiveMaxPool2d
from .sequence import *


POOL2D_LAYERS = {
    'gem': GeM(p=3.0, dim=2),
    'concat': AdaptiveConcatPool2d(),
    'avg': AdaptiveAvgPool2d(1),
    'avg2': AdaptiveAvgPool25d(),
    'max': AdaptiveMaxPool2d(1), 
}


def change_num_input_channels(model, in_channels=1):
    for i, m in enumerate(model.modules()):
      if isinstance(m, (nn.Conv2d,nn.Conv3d)) and m.in_channels == 3:
        m.in_channels = in_channels
        # First, sum across channels
        W = m.weight.sum(1, keepdim=True)
        # Then, divide by number of channels
        W = W / in_channels
        # Then, repeat by number of channels
        size = [1] * W.ndim
        size[1] = in_channels
        W = W.repeat(size)
        m.weight = nn.Parameter(W)
        break
    return model


def swap_pooling_layer(backbone, pool_layer_name, pool_layer):
    if hasattr(backbone, pool_layer_name):
        setattr(backbone, pool_layer_name, pool_layer)
    else:
        assert hasattr(backbone.head, pool_layer_name)
        setattr(backbone.head, pool_layer_name, pool_layer)
    return backbone


class Net2D(nn.Module):

    def __init__(self,
                 backbone,
                 pretrained,
                 num_classes,
                 dropout,
                 feat_reduce=None,
                 in_channels=3,
                 backbone_params={},
                 multisample_dropout=False,
                 pool=None,
                 load_pretrained=None,
                 freeze_backbone=False):
        super().__init__()
        self.backbone, dim_feats = backbones.get_backbone(name=backbone, pretrained=pretrained, **backbone_params)
        if pool and "swin" not in backbone:
            self.backbone = swap_pooling_layer(self.backbone, 
                POOLING[backbone], 
                POOL2D_LAYERS[pool])
        self.msdo = multisample_dropout
        if in_channels != 3: self.backbone = change_num_input_channels(self.backbone, in_channels)
        self.dropout = nn.Dropout(p=dropout)
        if isinstance(feat_reduce, int):
            self.feat_reduce = nn.Linear(dim_feats, feat_reduce)
            self.linear = nn.Linear(feat_reduce, num_classes)
        else:
            self.linear = nn.Linear(dim_feats, num_classes)
        if load_pretrained:
            print(f'Loading pretrained weights from {load_pretrained} ...')
            weights = torch.load(load_pretrained, map_location=lambda storage, loc: storage)['state_dict']
            weights = {re.sub(r'^model.', '', k) : v for k,v in weights.items()}
            # Get encoder only
            weights = {re.sub(r'^encoder.', '', k) : v for k,v in weights.items() if 'encoder' in k}
            self.backbone.load_state_dict(weights)
        if freeze_backbone:
            print('Freezing backbone ...')
            for param in self.backbone.parameters():
                param.requires_grad = False

    def extract_features(self, x):
        features = self.backbone(x)
        #features = features.view(features.shape[:2])
        if hasattr(self, 'feat_reduce'):
            features = self.feat_reduce(features)
        return features

    def forward(self, x):
        features = self.extract_features(x)
        if self.msdo:
            x = torch.mean(torch.stack([self.linear(self.dropout(features)) for _ in range(self.msdo)]), dim=0)
        else:
            x = self.linear(self.dropout(features))
        return x[:,0] if self.linear.out_features == 1 else x


class ConvSeq2D(nn.Module):

    def __init__(self,
                 backbone,
                 pretrained,
                 num_classes,
                 dropout,
                 num_slices=32,
                 num_conv_seq_layers=3,
                 feat_reduce=None,
                 in_channels=3,
                 backbone_params={},
                 multisample_dropout=False,
                 pool=None,
                 load_pretrained=None,
                 freeze_backbone=False):
        super().__init__()
        self.backbone, dim_feats = backbones.get_backbone(name=backbone, pretrained=pretrained, **backbone_params)
        if pool:
            self.backbone = swap_pooling_layer(self.backbone, 
                POOLING[backbone], 
                POOL2D_LAYERS[pool])
        self.msdo = multisample_dropout
        if in_channels != 3: self.backbone = change_num_input_channels(self.backbone, in_channels)
        self.dropout = nn.Dropout(p=dropout)
        if isinstance(feat_reduce, int):
            self.feat_reduce = nn.Linear(dim_feats, feat_reduce)
            dim_feats = feat_reduce
        if load_pretrained:
            print(f'Loading pretrained weights from {load_pretrained} ...')
            weights = torch.load(load_pretrained, map_location=lambda storage, loc: storage)['state_dict']
            weights = {re.sub(r'^model.', '', k) : v for k,v in weights.items()}
            # Get encoder only
            weights = {re.sub(r'^encoder.', '', k) : v for k,v in weights.items() if 'encoder' in k}
            self.backbone.load_state_dict(weights)
        if freeze_backbone:
            print('Freezing backbone ...')
            for param in self.backbone.parameters():
                param.requires_grad = False
        # ConvSeq head
        conv_seq = [] 
        for i in range(num_conv_seq_layers):
            conv = nn.Conv1d(dim_feats, dim_feats, kernel_size=3, stride=1, padding=1, bias=False)
            bn = nn.BatchNorm1d(dim_feats)
            act = nn.SiLU() 
            conv_seq.extend([conv, bn, act])
        self.conv_seq = nn.Sequential(*conv_seq)
        #self.slice_linear = nn.Linear(num_slices, 1)
        #self.linear = nn.Linear(dim_feats, num_classes)
        self.linear = nn.Linear(dim_feats, num_classes)

    def extract_features(self, x):
        features = self.backbone(x)
        features = features.view(features.shape[:2])
        if hasattr(self, 'feat_reduce'):
            features = self.feat_reduce(features)
        return features

    def extract_slice_features(self, x):
        # x.shape = (N, C, Z, H, W)
        features = torch.stack([self.extract_features(x[:,:,i]) for i in range(x.size(2))], dim=2)
        return features 

    def forward_head(self, x):
        features = self.conv_seq(x)
        features = features.mean(-1)
        if self.msdo:
            x = torch.mean(torch.stack([self.linear(self.dropout(features)) for _ in range(self.msdo)]), dim=0)
        else:
            x = self.linear(self.dropout(features))
        return x[:,0] if self.linear.out_features == 1 else x

    def forward(self, x):
        features = self.extract_slice_features(x)
        return self.forward_head(features)


class FuseNet2D(Net2D):

    def forward(self, x):
        # x.shape = (B, C, Z, H, W)
        features = torch.stack([self.extract_features(x[:,:,i]) for i in range(x.size(2))], axis=2)
        features = features.max(2)[0]
        if self.msdo:
            x = torch.mean(torch.stack([self.linear(self.dropout(features)) for _ in range(self.msdo)]), dim=0)
        else:
            x = self.linear(self.dropout(features))
        return x[:,0] if self.linear.out_features == 1 else x


MODEL_3D = ['ip_csn_50', 'ir_csn_50', 'mc3_18', 'r3d_18', 'r2plus1d_18', 'i3d', 'r2plus1d_34']

class Net3D(Net2D):

    def __init__(self, *args, **kwargs):
        no_stride = kwargs.pop('no_stride', False)
        super().__init__(*args, **kwargs)
        if kwargs['backbone'] not in MODEL_3D:
            self.backbone = backbones.convert_model_to_3d(self.backbone, no_stride=no_stride)

    def forward(self, x):
        features = self.extract_features(x)
        features = features.amax(dim=(-1, -2))
        features = features.mean(-1)
        if self.msdo:
            x = torch.mean(torch.stack([self.linear(self.dropout(features)) for _ in range(self.msdo)]), dim=0)
        else:
            x = self.linear(self.dropout(features))
        return x[:,0] if self.linear.out_features == 1 else x


class X3D(nn.Module):

    def __init__(self,
                 name="x3d_s",
                 pretrained=True,
                 num_classes=1,
                 dropout=0.2):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/pytorchvideo", model=name, pretrained=pretrained)
        self.backbone.blocks[-1].pool.pool = nn.Identity()
        del self.backbone.blocks[-1].dropout 
        del self.backbone.blocks[-1].proj 
        del self.backbone.blocks[-1].activation 
        del self.backbone.blocks[-1].output_pool 
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(self.backbone.blocks[-1].pool.post_conv.out_channels, num_classes)

    def forward(self, x):
        for idx in range(len(self.backbone.blocks) - 1):
            x = self.backbone.blocks[idx](x)
        x = self.backbone.blocks[-1].pool(x)
        x = x.mean((-1, -2, -3))
        x = self.dropout(x)
        x = self.linear(x)
        return x[:,0] if self.linear.out_features == 1 else x


class Net3DSlice(Net2D): 

    def __init__(self, *args, **kwargs):
        num_slices = kwargs.pop('num_slices', 32)
        super().__init__(*args, **kwargs)
        assert not hasattr(self, 'feat_reduce')
        if kwargs['backbone'] not in MODEL_3D:
            self.backbone = backbones.convert_model_to_3d(self.backbone, no_stride=True)
        self.backbone.head.global_pool = AdaptiveAvgPool25d()
        self.final_linear = nn.Linear(num_slices, 1)

    def extract_features(self, x):
        features = self.backbone(x)
        #features = features.view(features.shape[:2])
        if hasattr(self, 'feat_reduce'):
            features = self.feat_reduce(features)
        return features

    def forward(self, x):
        features = self.extract_features(x).transpose(-2, -1)
        if self.msdo:
            x = torch.mean(torch.stack([self.linear(self.dropout(features)) for _ in range(self.msdo)]), dim=0)
        else:
            x = self.linear(self.dropout(features))
        x = self.final_linear(x.transpose(-2, -1))
        x = x.view(x.shape[0], -1)
        return x[:,0] if self.linear.out_features == 1 else x


class SiameseNet3D(Net3D):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        df = self.linear.in_features
        self.head = nn.Linear(df*4, 1)
        del self.linear

    def forward_once(self, x):
        features = self.extract_features(x)
        return self.dropout(features)

    def forward(self, x): 
        x1, x2 = x
        feat1 = self.forward_once(x1)
        feat2 = self.forward_once(x2)
        l1 = torch.abs(feat1 - feat2)
        l2 = l1 ** 2
        ad = feat1 + feat2 
        mt = feat1 * feat2
        return self.head(torch.cat([l1, l2, ad, mt], dim=1))[:,0]


class Net2Dc(Net2D):

    def forward(self, x):
        x = x.squeeze(1)
        features = self.extract_features(x)
        if self.msdo:
            x = torch.mean(torch.stack([self.linear(self.dropout(features)) for _ in range(self.msdo)]), dim=0)
        else:
            x = self.linear(self.dropout(features))
        return x[:,0] if self.linear.out_features == 1 else x



class TDCNN(nn.Module):

    def __init__(self,
                 backbone,
                 pretrained,
                 num_classes,
                 in_channels=3,
                 load_transformer=None,
                 load_backbone=None,
                 feat_reduce=512,
                 pool='avg',
                 groups=1,
                 bn_eval=False,
                 backbone_params={},
                 transformer_params={}):
        super().__init__()
        bb = getattr(backbones, backbone)
        self.backbone, dim_feats = bb(pretrained=pretrained, **backbone_params)
        pool_layer = POOLING[backbone]
        pool_module = POOL2D_LAYERS[pool]
        if feat_reduce:
            pool_module = nn.Sequential(nn.Conv2d(dim_feats, feat_reduce, kernel_size=1, groups=groups), pool_module)
            dim_feats = feat_reduce
            self.feat_reduce = feat_reduce
        setattr(self.backbone, pool_layer, pool_module)
        if in_channels != 3: self.backbone = change_num_input_channels(self.backbone, in_channels)
        transformer_params['num_classes'] = num_classes
        self.transformer = TransformerCls(**transformer_params)

        if load_transformer:
            self.load_transformer(load_transformer)

        if load_backbone:
            self.load_backbone(load_backbone)

        self.bn_eval = bn_eval

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super().train(mode)
        if mode and self.bn_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()

    @staticmethod
    def _load_weights(fp):
        weights = torch.load(fp, map_location=lambda storage, loc: storage)
        weights = {re.sub(r'^module.', '', k) : v for k,v in weights.items()}
        return weights

    def load_backbone(self, fp):
        print(f'Loading pretrained backbone weights from {fp} ...')
        weights = self._load_weights(fp)
        weights = {k : v for k,v in weights.items() if re.search(r'^backbone.', k)}
        weights = {re.sub(r'^backbone.', '', k) : v for k,v in weights.items()}
        self.backbone.load_state_dict(weights)

    def load_transformer(self, fp):
        print(f'Loading pretrained transformer weights from {fp} ...')
        weights = self._load_weights(fp)
        weights = {k : v for k,v in weights.items() if re.search(r'^transformer.', k)} 
        weights = {re.sub(r'^transformer.', '', k) : v for k,v in weights.items()}
        self.transformer.transformer.load_state_dict(weights)

    def forward(self, x):
        #features = torch.stack([self.backbone(x[:,:,i]) for i in range(Z)], dim=1)
        #features = self.combine(features.transpose(1,2)).squeeze(-1)
        B, Z, C, H, W = x.size()
        x = x.view(B*Z, C, H, W)
        features = self.backbone(x)
        features = features.view(B, Z, -1)
        mask = torch.from_numpy(np.ones((B,features.size(1)))).long().to(features.device)
        # features.shape = (B, Z, embedding_dim)
        output = self.transformer((features,mask))
        return output


def NetSMP(model_type, encoder, **kwargs):
    return getattr(smp, model_type)(encoder, **kwargs)


from .smp.base.heads import SegmentationHead3d
from .smp.deeplabv3.decoder import DeepLabV3PlusDecoder


class MC3DeepLabV3Plus(nn.Module):

    def __init__(self,
                 num_classes,
                 in_channels=3,
                 dropout=0.0,
                 pretrained=True):
        super().__init__()
        self.encoder, dim_feats = backbones.mc3_18(pretrained=pretrained)
        if in_channels != 3: self.encoder = change_num_input_channels(self.encoder, in_channels)
        self.encoder.layer4[0].downsample[0].stride = 1
        self.encoder.layer4[0].conv1[0].stride = 1
        self.decoder = DeepLabV3PlusDecoder([3, 64, 64, 128, 256, 512])
        self.seg_head = SegmentationHead3d(256, num_classes, dropout=dropout, upsampling=4)

    def forward_encoder(self, x):
        features = [x]
        x = self.encoder.stem(x)
        x = F.max_pool3d(x, (1,2,2))
        features.append(x)
        for i in range(1, 5):
            x = getattr(self.encoder, f'layer{i}')(x)
            features.append(x)
        return features 

    def forward(self, x):
        features = self.forward_encoder(x)
        x = self.decoder(*features)
        x = self.seg_head(x)
        return x


class TimeSformer(nn.Module):

    def __init__(self,
                 num_classes,
                 in_channels=3,
                 dropout=0.0,
                 pretrained=True):
        super().__init__()
        self.model = _TimeSformer(
            img_size=224, 
            num_classes=1, 
            num_frames=48, 
            attention_type='divided_space_time',  
            pretrained_model='/home/ianpan/timesformer.pth', 
            dropout=dropout)
        if in_channels != 3:
            conv = self.model.model.patch_embed.proj
            conv.in_channels = in_channels
            w = conv.weight
            w = w.sum(1) / in_channels
            w = w.unsqueeze(1).repeat(1, in_channels, 1, 1)
            conv.weight = nn.Parameter(w)
            self.model.model.patch_embed.proj = conv

    def forward(self, x):
        return self.model(x)[:,0]


class EffB0_DeepLab(nn.Module):

    def __init__(self, 
                 num_classes,
                 in_channels=3,
                 dropout=0.0,
                 pretrained=True,
                 load_pretrained=None,
                 freeze_backbone=False,
                 add_classifier=False):
        super().__init__()
        self.encoder, dim_feats = backbones.efficientnet_b0_3d(pretrained=pretrained)
        if in_channels != 3: self.encoder = change_num_input_channels(self.encoder, in_channels)
        self.encoder.conv_stem.stride = 2
        self.encoder.blocks[-5][0].conv_dw.stride = 2
        self.encoder.blocks[-4][0].conv_dw.stride = 2
        self.encoder.blocks[-2][0].conv_dw.stride = 1
        self.encoder = nn.ModuleList([
            nn.Identity(),
            nn.Sequential(self.encoder.conv_stem, self.encoder.bn1, self.encoder.act1),
            self.encoder.blocks[:2],
            self.encoder.blocks[2:3],
            self.encoder.blocks[3:5],
            self.encoder.blocks[5:],
        ])
        self.add_classifier = add_classifier
        self.decoder = DeepLabV3PlusDecoder([3, 32, 24, 40, 112, 320])
        self.seg_head = SegmentationHead3d(256, num_classes, dropout=dropout, upsampling=4)  
        
        if self.add_classifier:
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(320+112+40+24, 1)

        if load_pretrained:
            print(f'Loading pretrained weights from {load_pretrained} ...')
            weights = torch.load(load_pretrained, map_location=lambda storage, loc: storage)['state_dict']
            weights = {re.sub(r'^model.', '', k) : v for k,v in weights.items()}
            # Get encoder only
            weights = {re.sub(r'^encoder.', '', k) : v for k,v in weights.items() if 'encoder' in k}
            self.encoder.load_state_dict(weights)

        if freeze_backbone:
            print('Freezing backbone ...')
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward_encoder(self, x):
        features = []
        for stage in self.encoder:
            x = stage(x)
            features.append(x)
        return features

    def forward(self, x):
        features = self.forward_encoder(x)
        x = self.decoder(*features)
        seg_out = self.seg_head(x)
        if self.add_classifier:
            x = torch.cat([F.adaptive_avg_pool3d(_, 1) for _ in features[2:]], dim=1)
            x = x.view(x.size(0), x.size(1))
            x = self.dropout(x)
            x = self.classifier(x)
            return seg_out, x[:,0] if self.classifier.out_features == 1 else x
        return seg_out
