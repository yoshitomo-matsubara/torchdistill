# Examples
## 1. ImageNet (ILSVRC 2012): Image Classification
### 1.1 Download the datasets
As the terms of use do not allow to distribute the URLs, you will have to create an account [here](http://image-net.org/download) to get the URLs, and replace `${TRAIN_DATASET_URL}` and `${VAL_DATASET_URL}` with them.
```
wget ${TRAIN_DATASET_URL} ./
wget ${VAL_DATASET_URL} ./
```

### 1.2 Untar and extract files
```
# Go to the root of this repository
mkdir ./resource/dataset/ilsvrc2012/{train,val} -p
mv ILSVRC2012_img_train.tar ./resource/dataset/ilsvrc2012/train/
cd ./resource/dataset/ilsvrc2012/train/
tar -xvf ILSVRC2012_img_train.tar
mv ILSVRC2012_img_train.tar ../
for f in *.tar; do
  d=`basename $f .tar`
  mkdir $d
  (cd $d && tar xf ../$f)
done
rm -r *.tar

mv ILSVRC2012_img_val.tar ./resource/dataset/ilsvrc2012/val/
wget https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
mv valprep.sh ./resource/dataset/ilsvrc2012/val/
cd ./resource/dataset/ilsvrc2012/val/
tar -xvf ILSVRC2012_img_val.tar
mv ILSVRC2012_img_val.tar ../
sh valprep.sh
mv valprep.sh ../
cd ../../../../
```

### 1.3 Run an experiment
e.g., Teacher: ResNet-152, Student: AlexNet  
a) Use GPUs for multiple distributed training processes
```
python3 -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --use_env examples/image_classification.py --world_size ${NUM_GPUS} --config config/sample/ilsvrc2012/single_stage/kd/alexnet_from_resnet152.yaml --log log/ilsvrc2012/kd/alexnet_from_resnet152.txt
```
b) Use GPU(s) for single training process
```
python3 examples/image_classification.py --config config/sample/ilsvrc2012/single_stage/kd/alexnet_from_resnet152.yaml --log log/ilsvrc2012/kd/alexnet_from_resnet152.txt
```  
c) Use CPU
```
python3 examples/image_classification.py --device cpu --config config/sample/ilsvrc2012/single_stage/kd/alexnet_from_resnet152.yaml --log log/ilsvrc2012/kd/alexnet_from_resnet152.txt
```  


## 2. COCO 2017: Object Detection
### 2.1 Download the datasets
```
wget http://images.cocodataset.org/zips/train2017.zip ./
wget http://images.cocodataset.org/zips/val2017.zip ./
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip ./
```

### 2.2 Unzip and extract files
```
# Go to the root of this repository
mkdir ./resource/dataset/coco2017/ -p
mv train2017.zip ./resource/dataset/coco2017/
mv val2017.zip ./resource/dataset/coco2017/
mv annotations_trainval2017.zip ./resource/dataset/coco2017/
cd ./resource/dataset/coco2017/
unzip train2017.zip
unzip val2017.zip
unzip annotations_trainval2017.zip
cd ../../../
```

### 2.3 Run an experiment
e.g., Teacher: Faster R-CNN with ResNet-50-FPN backbone, Student: Faster R-CNN with ResNet-18-FPN backbone  
a) Use GPUs for multiple distributed training processes
```
python3 -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --use_env examples/object_detection.py --world_size ${NUM_GPUS} --config config/sample/coco2017/multi_stage/ft/custom_fasterrcnn_resnet18_fpn_from_fasterrcnn_resnet50_fpn.yaml --log log/coco2017/ft/custom_fasterrcnn_resnet18_fpn_from_fasterrcnn_resnet50_fpn.txt
```
b) Use GPU(s) for single training process
```
python3 examples/object_detection.py --config config/sample/coco2017/multi_stage/ft/custom_fasterrcnn_resnet18_fpn_from_fasterrcnn_resnet50_fpn.yaml --log log/coco2017/ft/custom_fasterrcnn_resnet18_fpn_from_fasterrcnn_resnet50_fpn.txt
```  
c) Use CPU
```
python3 examples/object_detection.py --device cpu --config config/sample/coco2017/multi_stage/ft/custom_fasterrcnn_resnet18_fpn_from_fasterrcnn_resnet50_fpn.yaml --log log/coco2017/ft/custom_fasterrcnn_resnet18_fpn_from_fasterrcnn_resnet50_fpn.txt
```  
