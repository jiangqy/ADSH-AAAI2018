
---
#  Source code for ADSH-AAAI2018 [Matlab Version]
---
## Introduction
### 1. Running Environment
Matlab 2016

[MatConvnet](http://www.vlfeat.org/matconvnet/)
### 2. Datasets
We use three datasets to perform our experiments, i.e., CIFAR-10, MS-COCO and NUS-WIDE. You can preprocess these datasets by yourself or download from the following links:

[CIFAR-10 MAT File](http://pan.baidu.com/s/1miMgd7q)

[MS-COCO MAT File]()

[NUS-WIDE MAT File]()

In addition, pretrained model can be download from the following links:

[VGG-F](http://pan.baidu.com/s/1slhusrF)

### 3. Run demo
First you need download (or prepross by yourself) coresponding data and pretrained model and put them in the "data" folder. Then complie and run setup.m to configure MatConvNet.
Then run ADSH_demo().
```matlab
ADSH_demo
```