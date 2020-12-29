#!/bin/bash

set -xe
if [ ! -f test-data.tar.xz ] ;then
    wget -c https://github.com/junjihashimoto/hasktorch-yolo/releases/download/assets/test-data.tar.xz
fi
if [ ! -f weights.tar.xz ] ;then
    wget -c https://github.com/junjihashimoto/hasktorch-yolo/releases/download/assets/weights.tar.xz
fi

if [ ! -d test-data ] ;then
    tar xvfJ test-data.tar.xz
fi

if [ ! -d weights ] ;then
    tar xvfJ weights.tar.xz
fi
