from torch.hub import load_state_dict_from_url
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.keypoint_rcnn import KeypointRCNN
from torchvision.models.detection.mask_rcnn import MaskRCNN
from torchvision.ops import MultiScaleRoIAlign

from .resnet_backbone import custom_resnet_fpn_backbone
from ....registry import register_model_func


@register_model_func
def custom_fasterrcnn_resnet_fpn(backbone, pretrained=True, progress=True,
                                 num_classes=91, pretrained_backbone=True, trainable_backbone_layers=3, **kwargs):
    backbone_name = backbone['name']
    backbone_kwargs = backbone['kwargs']
    assert 0 <= trainable_backbone_layers <= 5
    # dont freeze any layers if pretrained model or backbone is not used
    if not (pretrained or pretrained_backbone):
        backbone_kwargs['trainable_backbone_layers'] = 5
    if pretrained:
        # no need to download the backbone if pretrained is set
        backbone_kwargs['pretrained'] = False

    backbone_model = custom_resnet_fpn_backbone(backbone_name, backbone_kwargs)
    num_feature_maps = len(backbone_model.body.return_layers)
    box_roi_pool = None if num_feature_maps == 4 \
        else MultiScaleRoIAlign(featmap_names=[str(i) for i in range(num_feature_maps)],
                                output_size=7, sampling_ratio=2)
    model = FasterRCNN(backbone_model, num_classes, box_roi_pool=box_roi_pool, **kwargs)
    if pretrained and backbone_name.endswith('resnet50'):
        state_dict = \
            load_state_dict_from_url('https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth',
                                     progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model


@register_model_func
def custom_maskrcnn_resnet_fpn(backbone, pretrained=True, progress=True,
                               num_classes=91, pretrained_backbone=True, trainable_backbone_layers=3, **kwargs):
    backbone_name = backbone['name']
    backbone_kwargs = backbone['kwargs']
    assert 0 <= trainable_backbone_layers <= 5
    # dont freeze any layers if pretrained model or backbone is not used
    if not (pretrained or pretrained_backbone):
        backbone_kwargs['trainable_backbone_layers'] = 5
    if pretrained:
        # no need to download the backbone if pretrained is set
        backbone_kwargs['pretrained'] = False

    backbone_model = custom_resnet_fpn_backbone(backbone_name, backbone_kwargs)
    num_feature_maps = len(backbone_model.body.return_layers)
    box_roi_pool = None if num_feature_maps == 4 \
        else MultiScaleRoIAlign(featmap_names=[str(i) for i in range(num_feature_maps)],
                                output_size=7, sampling_ratio=2)
    mask_roi_pool = None if num_feature_maps == 4 \
        else MultiScaleRoIAlign(featmap_names=[str(i) for i in range(num_feature_maps)],
                                output_size=14, sampling_ratio=2)
    model = MaskRCNN(backbone_model, num_classes, box_roi_pool=box_roi_pool, mask_roi_pool=mask_roi_pool **kwargs)
    if pretrained and backbone_name.endswith('resnet50'):
        state_dict = \
            load_state_dict_from_url('https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth',
                                     progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model


@register_model_func
def custom_keypointrcnn_resnet_fpn(backbone, pretrained=True, progress=True, num_classes=2, num_keypoints=17,
                                   pretrained_backbone=True, trainable_backbone_layers=3, **kwargs):
    backbone_name = backbone['name']
    backbone_kwargs = backbone['kwargs']
    assert 0 <= trainable_backbone_layers <= 5
    # dont freeze any layers if pretrained model or backbone is not used
    if not (pretrained or pretrained_backbone):
        backbone_kwargs['trainable_backbone_layers'] = 5
    if pretrained:
        # no need to download the backbone if pretrained is set
        backbone_kwargs['pretrained'] = False

    backbone_model = custom_resnet_fpn_backbone(backbone_name, backbone_kwargs)
    num_feature_maps = len(backbone_model.body.return_layers)
    box_roi_pool = None if num_feature_maps == 4 \
        else MultiScaleRoIAlign(featmap_names=[str(i) for i in range(num_feature_maps)],
                                output_size=7, sampling_ratio=2)
    keypoint_roi_pool = None if num_feature_maps == 4 \
        else MultiScaleRoIAlign(featmap_names=[str(i) for i in range(num_feature_maps)],
                                output_size=14, sampling_ratio=2)
    model = KeypointRCNN(backbone_model, num_classes, num_keypoints=num_keypoints, box_roi_pool=box_roi_pool,
                         keypoint_roi_pool=keypoint_roi_pool, **kwargs)
    if pretrained and backbone_name.endswith('resnet50'):
        state_dict = \
            load_state_dict_from_url('https://download.pytorch.org/models/keypointrcnn_resnet50_fpn_coco-fc266e95.pth',
                                     progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model
