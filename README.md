# 基于DenseNet轻量化的板式换热器板片缺陷检测方法研究

## 项目简介
本项目基于DenseNet等深度学习网络，针对缺陷检测任务，提供了完整的训练、推理、评估和可视化流程，适用于工业缺陷检测领域的研究与应用。
训练模型包括基线模型（resnet50,efficientnet_b0,shufflenetv2）
           论文17模型，论文20模型(densenet_literature17,densenet_literature20)
           Densenet基础模型(densenet121yuan,densenet169)
           针对下采样与卷积核尺寸的对比实验的模型(densenet_mod1 , densenet_mod2, densenet_mod3,densenet_mod4 ,densenet_mod5)，
           本论文根据感受野缺陷尺寸实例化的模型为(densenet_mod6)

## 主要功能
- 支持DenseNet、ResNet等多种主干网络的缺陷检测模型训练与推理
- VOC格式数据集的处理与评估
- 检测结果可视化
- 多GPU训练支持

## 环境依赖
请先安装requirements.txt中的所有依赖：
```bash
pip install -r requirements.txt
```

## 快速开始
1. 数据准备：将数据集按VOC格式组织。
2. 训练模型：
```bash
python train_desnet.py
```
3. 推理与评估：
```bash
python defect_instance_evaluation_v2.py
```
4. 可视化：
```bash
python visualize_defect_instances_cv2.py
```

## 主要文件结构
- `backbone/`：主干网络相关代码
- `network_files/`：Faster R-CNN等检测网络实现
- `train_utils/`：训练与评估工具
- `visualization_results_cv2/`：可视化结果
- 其余py文件：数据处理、训练、推理、评估等脚本

## 注意事项
- 本项目未包含任何权重文件、数据集、标签等，仅保留代码。
- 若需训练或推理，请自行准备数据集和预训练权重。
- 建议使用Python 3.7及以上版本。

## 贡献与致谢
欢迎提出issue或pull request改进本项目。

---
如有问题请联系作者或在GitHub提交issue。 