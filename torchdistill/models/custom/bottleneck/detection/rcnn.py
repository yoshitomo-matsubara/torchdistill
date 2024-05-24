from torch.hub import load_state_dict_from_url
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.keypoint_rcnn import KeypointRCNN
from torchvision.models.detection.mask_rcnn import MaskRCNN
from torchvision.ops import MultiScaleRoIAlign

from .resnet_backbone import custom_resnet_fpn_backbone
from ....registry import register_model


@register_model
def custom_fasterrcnn_resnet_fpn(backbone, weights=None, progress=True,
                                 num_classes=91, trainable_backbone_layers=3, **kwargs):
    backbone_key = backbone['key']
    backbone_kwargs = backbone['kwargs']
    assert 0 <= trainable_backbone_layers <= 5
    # don't freeze any layers if pretrained model or backbone is not used
    if weights is not None and 'trainable_backbone_layers' not in backbone_kwargs:
        backbone_kwargs['trainable_backbone_layers'] = 5

    backbone_model = custom_resnet_fpn_backbone(backbone_key, backbone_kwargs)
    num_feature_maps = len(backbone_model.body.return_layers)
    box_roi_pool = None if num_feature_maps == 4 \
        else MultiScaleRoIAlign(featmap_names=[str(i) for i in range(num_feature_maps)],
                                output_size=7, sampling_ratio=2)
    model = FasterRCNN(backbone_model, num_classes, box_roi_pool=box_roi_pool, **kwargs)
    if weights is not None:
        state_dict = \
            load_state_dict_from_url(weights.url, progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model


@register_model
def custom_maskrcnn_resnet_fpn(backbone, weights=None, progress=True,
                               num_classes=91, trainable_backbone_layers=3, **kwargs):
    backbone_key = backbone['key']
    backbone_kwargs = backbone['kwargs']
    assert 0 <= trainable_backbone_layers <= 5
    # don't freeze any layers if pretrained model or backbone is not used
    if weights is not None and 'trainable_backbone_layers' not in backbone_kwargs:
        backbone_kwargs['trainable_backbone_layers'] = 5

    backbone_model = custom_resnet_fpn_backbone(backbone_key, backbone_kwargs)
    num_feature_maps = len(backbone_model.body.return_layers)
    box_roi_pool = None if num_feature_maps == 4 \
        else MultiScaleRoIAlign(featmap_names=[str(i) for i in range(num_feature_maps)],
                                output_size=7, sampling_ratio=2)
    mask_roi_pool = None if num_feature_maps == 4 \
        else MultiScaleRoIAlign(featmap_names=[str(i) for i in range(num_feature_maps)],
                                output_size=14, sampling_ratio=2)
    model = MaskRCNN(backbone_model, num_classes, box_roi_pool=box_roi_pool, mask_roi_pool=mask_roi_pool, **kwargs)
    if weights is not None:
        state_dict = \
            load_state_dict_from_url(weights.url, progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model


@register_model
def custom_keypointrcnn_resnet_fpn(backbone, weights=None, progress=True, num_classes=2, num_keypoints=17,
                                   trainable_backbone_layers=3, **kwargs):
    backbone_key = backbone['key']
    backbone_kwargs = backbone['kwargs']
    assert 0 <= trainable_backbone_layers <= 5
    # don't freeze any layers if pretrained model or backbone is not used
    if weights is not None and 'trainable_backbone_layers' not in backbone_kwargs:
        backbone_kwargs['trainable_backbone_layers'] = 5

    backbone_model = custom_resnet_fpn_backbone(backbone_key, backbone_kwargs)
    num_feature_maps = len(backbone_model.body.return_layers)
    box_roi_pool = None if num_feature_maps == 4 \
        else MultiScaleRoIAlign(featmap_names=[str(i) for i in range(num_feature_maps)],
                                output_size=7, sampling_ratio=2)
    keypoint_roi_pool = None if num_feature_maps == 4 \
        else MultiScaleRoIAlign(featmap_names=[str(i) for i in range(num_feature_maps)],
                                output_size=14, sampling_ratio=2)
    model = KeypointRCNN(backbone_model, num_classes, num_keypoints=num_keypoints, box_roi_pool=box_roi_pool,
                         keypoint_roi_pool=keypoint_roi_pool, **kwargs)
    if weights is not None:
        state_dict = \
            load_state_dict_from_url(weights.url, progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model
