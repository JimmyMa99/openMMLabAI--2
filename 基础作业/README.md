## 基础实验
请参考 MMDetection 文档及教程，基于自定义数据集 balloon 训练实例分割模型，基于训练的模型在样例视频上完成color splash的效果制作，即使用模型对图像进行逐帧实例分割，并将气球以外的图像转换为灰度图像。

https://github.com/open-mmlab/OpenMMLabCamp/blob/main/AI%20%E5%AE%9E%E6%88%98%E8%90%A5%E5%9F%BA%E7%A1%80%E7%8F%AD/%E4%BD%9C%E4%B8%9A%E4%BA%8C_mmdetection.md

## 实验设备
NVIDIA GeForce GTX 1080 Ti

##  Balloon数据集

#### 数据集介绍

balloon是带有mask的气球数据集，其中训练集包含61张图片，验证集包含13张图片。

下载链接：https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip



### balloon

|        Model        |  bbox_mAP (%) |
| :-----------------: |  :-------: |
| mask_rcnn(r50fpn_fp16_1x) |   76.58   |
### 更新
看了看同学的脚本，写得比我好，我更新一下

https://github.com/aso538/OpenMMLab_AI_camp_work/blob/main/basic_wor_2/color_splash.py
 
checkpoints link: 链接：https://pan.baidu.com/s/1pIos1OEFuJSTX6prDxBhSQ 提取码：gbpy 

vedio link: 链接：https://pan.baidu.com/s/1pT2J4fNab7kMlIdiYd0Lrg 提取码：0hn1 

- gif效果图

![效果图](https://github.com/JimmyMa99/openMMLabAI--2/blob/e038cbf83c276429480daea3a799ec163b8c87e3/%E5%9F%BA%E7%A1%80%E4%BD%9C%E4%B8%9A/color_splash%2000_00_00-00_00_30.gif)
