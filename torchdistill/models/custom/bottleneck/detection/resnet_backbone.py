from torch import nn
from torchvision.models import resnet
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.ops import misc as misc_nn_ops

from torchdistill.models.custom.bottleneck.base import BottleneckBase
from torchdistill.models.custom.bottleneck.processor import get_bottleneck_processor


class Bottleneck4SmallResNet(BottleneckBase):
    def __init__(self, bottleneck_channel, compressor=None, decompressor=None):
        encoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 256, kernel_size=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, kernel_size=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, bottleneck_channel, kernel_size=2, padding=1, bias=False)
        )
        decoder = nn.Sequential(
            nn.BatchNorm2d(bottleneck_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channel, 64, kernel_size=2, bias=False),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=2, bias=False),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        super().__init__(encoder=encoder, decoder=decoder, compressor=compressor, decompressor=decompressor)


class Bottleneck4LargeResNet(BottleneckBase):
    def __init__(self, bottleneck_channel, compressor=None, decompressor=None):
        encoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 256, kernel_size=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, kernel_size=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, bottleneck_channel, kernel_size=2, padding=1, bias=False)
        )
        decoder = nn.Sequential(
            nn.BatchNorm2d(bottleneck_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channel, 64, kernel_size=2, bias=False),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=2, bias=False),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        super().__init__(encoder=encoder, decoder=decoder, compressor=compressor, decompressor=decompressor)

    def get_ext_classifier(self):
        return self.encoder.get_ext_classifier()


def custom_resnet_fpn_backbone(backbone_name, backbone_params_config,
                               norm_layer=misc_nn_ops.FrozenBatchNorm2d):
    layer1_config = backbone_params_config.get('layer1', None)
    layer1 = None
    if layer1_config is not None:
        compressor_config = layer1_config.get('compressor', None)
        compressor = None if compressor_config is None \
            else get_bottleneck_processor(compressor_config['name'], **compressor_config['params'])
        decompressor_config = layer1_config.get('decompressor', None)
        decompressor = None if decompressor_config is None \
            else get_bottleneck_processor(decompressor_config['name'], **decompressor_config['params'])

        layer1_type = layer1_config['type']
        if layer1_type == 'Bottleneck4SmallResNet' and backbone_name in {'custom_resnet18', 'custom_resnet34'}:
            layer1 = Bottleneck4SmallResNet(layer1_config['bottleneck_channel'], compressor, decompressor)
        elif layer1_type == 'Bottleneck4LargeResNet'\
                and backbone_name in {'custom_resnet50', 'custom_resnet101', 'custom_resnet152'}:
            layer1 = Bottleneck4LargeResNet(layer1_config['bottleneck_channel'], compressor, decompressor)

    prefix = 'custom_'
    start_idx = backbone_name.find(prefix) + len(prefix)
    org_backbone_name = backbone_name[start_idx:] if backbone_name.startswith(prefix) else backbone_name
    backbone = resnet.__dict__[org_backbone_name](
        pretrained=backbone_params_config.get('pretrained', False),
        norm_layer=norm_layer
    )
    if layer1 is not None:
        backbone.layer1 = layer1

    trainable_layers = backbone_params_config.get('trainable_backbone_layers', 3)
    # select layers that wont be frozen
    assert 0 <= trainable_layers <= 5
    layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:trainable_layers]
    # freeze layers only if pretrained backbone is used
    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}

    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [
        in_channels_stage2,
        in_channels_stage2 * 2,
        in_channels_stage2 * 4,
        in_channels_stage2 * 8,
    ]
    out_channels = 256
    return BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels)
