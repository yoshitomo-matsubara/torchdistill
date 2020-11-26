# Teacher: ResNet-34, Student: ResNet-18
## Knowledge Distillation
```
python3 examples/image_classification.py --config configs/official/ilsvrc2012/kd-resnet18_from_resnet34.yaml -test_only
```

## Attention Transfer
```
python3 examples/image_classification.py --config configs/official/ilsvrc2012/at-resnet18_from_resnet34.yaml -test_only
```

## Factor Transfer
```
python3 examples/image_classification.py --config configs/official/ilsvrc2012/ft-resnet18_from_resnet34.yaml -test_only
```

## Teacher-free Knowledge Distillation
```
python3 examples/image_classification.py --config configs/official/ilsvrc2012/tfkd-resnet18_from_resnet34.yaml -test_only
```

## Semi-supervisioned Knowledge Distillation
```
python3 examples/image_classification.py --config configs/official/ilsvrc2012/sskd-resnet18_from_resnet34.yaml -test_only
```

## L2 (CSE + L2)
```
python3 examples/image_classification.py --config configs/official/ilsvrc2012/cse_l2-resnet18_from_resnet34.yaml -test_only
```

## PAD-L2 (2nd stage)
```
python3 examples/image_classification.py --config configs/official/ilsvrc2012/pad-resnet18_from_resnet34.yaml -test_only
```

# Generalized Head Network Distillation
## Teacher: Faster R-CNN with ResNet-50 and FPN, Student: Faster R-CNN with
Bottleneck-injected ResNet-50 and FPN
```
python3 examples/object_detection.py --config configs/official/coco2017/ghnd_fasterrcnn.yaml -test_only
```

## Teacher: Mask R-CNN with ResNet-50 and FPN, Student: Mask R-CNN with
Bottleneck-injected ResNet-50 and FPN
```
python3 examples/object_detection.py --config configs/official/coco2017/ghnd_maskrcnn.yaml -test_only
```