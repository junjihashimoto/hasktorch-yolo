# Hasktorch Yolo

This repository develops yolov3 using hasktorch.
It is based on https://github.com/eriklindernoren/PyTorch-YOLOv3.

It supports the format of the original yolo config and weight files.
See [this link](https://pjreddie.com/darknet/yolo/) for original yolo.

For now, we use pre-trained weights.
We do not support testing and training, yet.

The sources of the config and weight file are following links.
These are already included in this repository.

```
https://raw.githubusercontent.com/eriklindernoren/PyTorch-YOLOv3/master/config/yolov3.cfg
https://pjreddie.com/media/files/yolov3.weights

```

# Getting Started

### linux+cabal

```shell
git clone git@github.com:junjihashimoto/hasktorch-yolo.git
cd hasktorch-yolo
./download-weights.sh
cabal test all
```

### linux+nix

```shell
git clone git@github.com:junjihashimoto/hasktorch-yolo.git
cd hasktorch-yolo
./download-weights.sh
nix-build
```

# Inference

Use the following command for inference.
An image with a bounding box is output.
The execution example is fig.1.

### linux+cabal

```shell
cabal run yolov3 -- config/yolov3.cfg weights/yolov3.weights test-data/train.jpg out.png
```

### linux+nix

```shell
nix-build
./result-2/bin/yolov3 config/yolov3.cfg weights/yolov3.weights test-data/train.jpg out.png
```

![fig.1](screenshot.png)


# Test

The command to calcurate mAP is as follows.
It supports both CPU and CUDA.

### CPU

```
DEVICE=cpu cabal run yolov3-pipelined-test --enable-profiling -- config/yolov3.cfg weights/yolov3.weights ./coco.data  +RTS -p -hc -N3
```

### DEVICE

```
DEVICE=cuda:0 cabal run yolov3-pipelined-test --enable-profiling -- config/yolov3.cfg weights/yolov3.weights ./coco.data  +RTS -p -hc -N3
```

# Training

The command to train a model is as follows.
It supports both CPU and CUDA.

### CPU

```
DEVICE=cpu cabal run yolov3-training --enable-profiling -- config/yolov3.cfg weights/yolov3.weights ./coco.data  +RTS -p -hc -N3
```

### DEVICE

```
DEVICE=cuda:0 cabal run yolov3-training --enable-profiling -- config/yolov3.cfg weights/yolov3.weights ./coco.data  +RTS -p -hc -N3
```
