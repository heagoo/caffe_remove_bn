# Caffe

[![Build Status](https://travis-ci.org/BVLC/caffe.svg?branch=master)](https://travis-ci.org/BVLC/caffe)
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

Caffe is a deep learning framework made with expression, speed, and modularity in mind.
It is developed by the Berkeley Vision and Learning Center ([BVLC](http://bvlc.eecs.berkeley.edu)) and community contributors.

This repository added a tool named caffe_remove_bn, which is designed to remove the BatchNorm and Scale layer inside the network topologies like ResNet, Inception v3/v4, etc. (Of course, ONLY for inference, as the inference uses the global mean and variance, which can be combined into the convolution layer)

It will take a trained model as input, and output a transformed model.

Usage example:
.build_release/tools/remove_bn_layer.bin ResNet-50-deploy.prototxt ResNet-50-model.caffemodel bn_removed.prototxt bn_removed.caffemodel

Please note: Many special cases are not considered in the implementation. Please modifiy the code if needed.

Happy brewing!

## Modifications
modified:   include/caffe/net.hpp
modified:   src/caffe/net.cpp
Added:      tools/remove_bn_layer.cpp

## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }
