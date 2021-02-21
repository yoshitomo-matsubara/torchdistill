from torchvision.models.detection.faster_rcnn import FasterRCNN, model_urls as fasterrcnn_model_urls
from torchvision.models.detection.keypoint_rcnn import KeypointRCNN, model_urls as keypointrcnn_model_urls
from torchvision.models.detection.mask_rcnn import MaskRCNN, model_urls as maskrcnn_model_urls
from torchvision.models.utils import load_state_dict_from_url
from torchvision.ops import MultiScaleRoIAlign

from torchdistill.models.custom.bottleneck.detection.resnet_backbone import custom_resnet_fpn_backbone
from torchdistill.models.registry import register_model_func


@register_model_func
def custom_fasterrcnn_resnet_fpn(backbone, pretrained=True, progress=True,
                                 num_classes=91, pretrained_backbone=True, trainable_backbone_layers=3, **kwargs):
    backbone_name = backbone['name']
    backbone_params_config = backbone['params']
    assert 0 <= trainable_backbone_layers <= 5
    # dont freeze any layers if pretrained model or backbone is not used
    if not (pretrained or pretrained_backbone):
        backbone_params_config['trainable_backbone_layers'] = 5
    if pretrained:
        # no need to download the backbone if pretrained is set
        backbone_params_config['pretrained'] = False

    backbone_model = custom_resnet_fpn_backbone(backbone_name, backbone_params_config)
    num_feature_maps = len(backbone_model.body.return_layers)
    box_roi_pool = None if num_feature_maps == 4 \
        else MultiScaleRoIAlign(featmap_names=[str(i) for i in range(num_feature_maps)],
                                output_size=7, sampling_ratio=2)
    model = FasterRCNN(backbone_model, num_classes, box_roi_pool=box_roi_pool, **kwargs)
    if pretrained and backbone_name.endswith('resnet50'):
        state_dict = load_state_dict_from_url(fasterrcnn_model_urls['fasterrcnn_resnet50_fpn_coco'], progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model


@register_model_func
def custom_maskrcnn_resnet_fpn(backbone, pretrained=True, progress=True,
                               num_classes=91, pretrained_backbone=True, trainable_backbone_layers=3, **kwargs):
    backbone_name = backbone['name']
    backbone_params_config = backbone['params']
    assert 0 <= trainable_backbone_layers <= 5
    # dont freeze any layers if pretrained model or backbone is not used
    if not (pretrained or pretrained_backbone):
        backbone_params_config['trainable_backbone_layers'] = 5
    if pretrained:
        # no need to download the backbone if pretrained is set
        backbone_params_config['pretrained'] = False

    backbone_model = custom_resnet_fpn_backbone(backbone_name, backbone_params_config)
    num_feature_maps = len(backbone_model.body.return_layers)
    box_roi_pool = None if num_feature_maps == 4 \
        else MultiScaleRoIAlign(featmap_names=[str(i) for i in range(num_feature_maps)],
                                output_size=7, sampling_ratio=2)
    mask_roi_pool = None if num_feature_maps == 4 \
        else MultiScaleRoIAlign(featmap_names=[str(i) for i in range(num_feature_maps)],
                                output_size=14, sampling_ratio=2)
    model = MaskRCNN(backbone_model, num_classes, box_roi_pool=box_roi_pool, mask_roi_pool=mask_roi_pool **kwargs)
    if pretrained and backbone_name.endswith('resnet50'):
        state_dict = load_state_dict_from_url(maskrcnn_model_urls['maskrcnn_resnet50_fpn_coco'], progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model


@register_model_func
def custom_keypointrcnn_resnet_fpn(backbone, pretrained=True, progress=True, num_classes=2, num_keypoints=17,
                                   pretrained_backbone=True, trainable_backbone_layers=3, **kwargs):
    backbone_name = backbone['name']
    backbone_params_config = backbone['params']
    assert 0 <= trainable_backbone_layers <= 5
    # dont freeze any layers if pretrained model or backbone is not used
    if not (pretrained or pretrained_backbone):
        backbone_params_config['trainable_backbone_layers'] = 5
    if pretrained:
        # no need to download the backbone if pretrained is set
        backbone_params_config['pretrained'] = False

    backbone_model = custom_resnet_fpn_backbone(backbone_name, backbone_params_config)
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
            load_state_dict_from_url(keypointrcnn_model_urls['keypointrcnn_resnet50_fpn_coco'], progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model
