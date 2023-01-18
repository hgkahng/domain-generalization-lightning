
import typing

import torch
import torch.nn as nn

from dg_lightning.networks.cnn import CNNForMNIST
from dg_lightning.networks.resnet import ResNetBackbone
from dg_lightning.networks.densenet import DenseNetBackbone
from dg_lightning.networks.bert import DistilBertBackbone



class NetworkInitializer(object):
    
    _supported_datasets: typing.List[str] = [
        'rmnist',
        'cmnist',
        'pacs',
        'vlcs',
        'celeba',
        'camelyon17',
        'povertymap',
        'iwildcam',
        'rxrx1',
        'fmow',
        'civilcomments',
    ]

    _in_channels: typing.Dict[str, int] = {
        'rmnist': 1,
        'cmnist': 3,
        'pacs': 3,
        'vlcs': 3,
        'celeba': 3,
        'camelyon17': 3,
        'povertymap': 8,
        'iwildcam': 3,
        'rxrx1': 3,
        'fmow': 3,
        'civilcomments': None,
    }

    _out_features: typing.Dict[str, typing.Tuple[int, str]] = {
        'rmnist': (10, 'multiclass'),
        'cmnist': (1, 'binary'),
        'pacs': (7, 'multiclass'),
        'vlcs': (5, 'multiclass'),
        'celeba': (1, 'binary'),
        'camelyon17': (1, 'binary'),
        'povertymap': (1, 'regression'),
        'iwildcam': (182, 'multiclass'),
        'rxrx1': (1139, 'multiclass'),
        'fmow': (62, 'multiclass'),
        'civilcomments': (1, 'binary'),
    }
    
    @classmethod
    def initialize_backbone(cls, name: str, data: str, pretrained: bool) -> nn.Module:
        """
        Helper function for initializing backbones (encoders).
        """
        if data not in cls._supported_datasets:
            raise ValueError(
                f"Invalid option for data (={data}). Supports {', '.join(cls._supported_datasets)}."
            )
        if data in ['civilcomments', 'amazon', ]:
            return cls.initialize_bert_backbone(name=name, data=data, pretrained=pretrained)
        elif data in ['cmnist', 'rmnist', ]:
            return CNNForMNIST(in_channels=cls._in_channels[data])
        else:
            return cls.initialize_cnn_backbone(name=name, data=data, pretrained=pretrained)

    @classmethod
    def initialize_cnn_backbone(cls, name: str, data: str, pretrained: bool) -> nn.Module:
        """
        Helper function for initializing CNN-based backbones.
            nn.Sequential(backbone, global_average_pooling, flatten)
        """
        in_channels: int = cls._in_channels[data]
        if name.startswith('resnet'):
            return ResNetBackbone(name=name, in_channels=in_channels, pretrained=pretrained)
        elif name.startswith('wideresnet'):
            raise NotImplementedError("Work in progress.")
        elif name.startswith('densenet'):
            return DenseNetBackbone(name=name, in_channels=in_channels, pretrained=pretrained)
        else:
            raise ValueError("Only supports {cnn, resnet, wideresnet, densenet}-based models.")

    @classmethod
    def initialize_bert_backbone(cls, name: str, **kwargs) -> nn.Module:
        if name not in ['distilbert-base-uncased', ]:
            raise NotImplementedError
        return DistilBertBackbone(name=name)

    @classmethod
    def initialize_linear_output_layer(cls, data: str, backbone: nn.Module, add_features: int = 0) -> nn.Module:
        """
        FIXME: this function is deprecated, and will be removed in future updates.
        Helper function for initializing output {classification, regression} layer.
        Arguments:
            data: str,
            backbone: nn.Module,
            add_features: int,
        Returns:
            nn.Linear
        """

        out_features, _ = cls._out_features[data]
        if isinstance(backbone, (ResNetBackbone, DenseNetBackbone)):
            in_features: int = backbone.out_features + add_features
            linear = nn.Linear(in_features, out_features, bias=True)
            linear.bias.data.fill_(0.)
            return linear
        else:
            raise NotImplementedError(f"Backbone: `{backbone.__class__.__name__}`")
