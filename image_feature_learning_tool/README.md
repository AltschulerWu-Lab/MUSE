# Learning deep image features with pretrained neural network

This is a demonstration to extract deep images features from Spatial Transcriptomics images using Google Inception-v3 pretrained on ImageNet.

The pretained Inception-v3 model was obtained from [Tensorflow Hub](https://tfhub.dev/). It takes in images with `299*299*3` size and output the last full connection layer with `2048` dimensions as features.

## Package requirement

- Tensorflow
- Tensorflow_hub
- Numpy
- Opencv-python
- Pandas


## How to use it?

In this demonstration, there are three file/folders:

```bash
# Pretrained neural network provided by Tensorflow Hub
inception-v3

# Two example inputs
example_img/
|_ Img_1.npy
|_ Img_2.npy

# Deep feature pipeline
image_inception.py
```
Segmentated images were save to `example_img` folder. `image_inception.py` will sequential load images from this folder, resize images then use model structure and parameters defined in `inception-v3` to do feed forward inference. Last full connection layer of the model is used as features.   

#### Step 1: Segment single cell or spot to patches

Use the `image + cell masks` (for imaging based single-cell spatial transcriptomics methods, e.g. seqFISH+, merFISH and STARmap) or `image + spot positions` (for sequencing based methods, e.g. Spatial Transcriptomics [ST], 10X Visium) to identify regions corresponding to each cell or spot and save them in individual files. The format can be either image or numpy array (`*.npy`). Segmented image size can be in any scale and the pipeline will automatically adjust to  the default input size of Inception-v3 model (`299 * 299`). 

All segmented images will be saved in `example_img` and named as `Img_#.npy` (or any image formats) to define the reading order. 


#### Step 2: Load segmented patches and learn corresponding image features

In python script `image_inception.py`, we load pretrained network from tensorflow hub, combine all segmented patches in Step 1 and input then to the network to obtain deep features.  

## Copyright
Software provided as is under **MIT License**.

Copyright (c) 2020 Altschuler and Wu Lab

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

