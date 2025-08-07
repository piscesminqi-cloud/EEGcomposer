# 0.环境配置

```shell
cd EEGComposer
pip install -r requirement.txt
```

# 1.图像预处理

本项目在COCO数据集上预训练，选用unlabeled2017

## 1.1首先下载COCO数据集并解压

```shell
mkdir data
wget http://images.cocodataset.org/zips/unlabeled2017.zip
unzip -d data/ unlabeled2017.zip
```

## 1.2计算颜色直方图

计算好颜色直方图后，保存到data文件夹下

```shell
python rayleigh-master/rayleigh/image2color.py
```

## 1.3获取图片对应的文字描述

将图片对应的文字描述存储在csv文件中，保存到data文件夹下

```shell
python image2text.py
```

## 1.4计算局部条件（草图、深度图、实例分割图、强度图）

首先下载对应模型的预训练权重（MiDaS、segment-anything）并复制到对应目录下，然后运行预处理脚本

```shell
wget https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_large_512.pt
cp dpt_beit_large_512.pt MiDaS_master/weights/
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
cp sam_vit_l_0b3195.pth segment_anything_main/
python preprocess.py
```

# 2.训练

运行TrainComposer.py开始训练

```shell
python TrainComposer.py
```

# 3.推理

运行Infer.py进行推理

或运行InferAll.py推理所有测试集图片

```shell
python Infer.py
```

