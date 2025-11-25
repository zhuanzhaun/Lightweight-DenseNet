import os
import datetime
import sys
import torch
import torchvision
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor
import transforms
from network_files import FasterRCNN, AnchorsGenerator
from my_dataset import VOCDataSet
from train_utils import GroupedBatchSampler, create_aspect_ratio_groups
from train_utils import train_eval_utils as utils
sys.setrecursionlimit(100000000)

# 尝试导入 efficientnet_lite_builder
USE_LITE_BUILDER = False
try:
    # 如果 efficientnet_lite_builder 是 PyTorch 版本，可以直接使用
    from efficientnet_lite_builder import build_model_base, efficientnet_lite_params
    USE_LITE_BUILDER = True
    print("成功导入 efficientnet_lite_builder")
except (ImportError, ModuleNotFoundError):
    # 如果导入失败，使用 timm 库的 EfficientNet-Lite0
    USE_LITE_BUILDER = False
    try:
        import timm
        print("使用 timm 库的 EfficientNet-Lite0")
    except ImportError:
        print("警告: 未找到 efficientnet_lite_builder 或 timm 库")
        print("请安装 timm: pip install timm")
        print("将使用 torchvision 的 EfficientNet-B0 作为替代")


def create_efficientnet_lite0_backbone():
    """
    创建 EfficientNet-Lite0 backbone
    优先使用 efficientnet_lite_builder，否则使用 timm
    """
    use_builder = USE_LITE_BUILDER
    
    if use_builder:
        # 使用 efficientnet_lite_builder（如果是 PyTorch 版本）
        try:
            # 尝试使用 builder
            width_coefficient, depth_coefficient, resolution, dropout_rate = efficientnet_lite_params('efficientnet-lite0')
            # 注意：这里需要根据实际的 efficientnet_lite_builder 实现来调整
            # 如果它是 TensorFlow 版本，我们需要使用 timm 替代
            print("注意: efficientnet_lite_builder 是 TensorFlow 版本，将使用 timm 替代")
            use_builder = False
        except Exception as e:
            # 如果 builder 不可用，使用 timm
            print(f"使用 efficientnet_lite_builder 失败: {e}，将使用 timm")
            use_builder = False
    
    if not use_builder:
        # 使用 timm 库的 EfficientNet-Lite0
        try:
            # timm 中的 EfficientNet-Lite0 模型名称
            model = timm.create_model('efficientnet_lite0', pretrained=True, features_only=True)
            # 获取特征提取层
            # EfficientNet-Lite0 的输出通道数通常是 1280（最后一层特征）
            # 我们需要找到合适的特征层
            with torch.no_grad():
                test_input = torch.randn(1, 3, 608, 608)
                # timm 的 features_only 模式会返回多个特征层
                features = model(test_input)
                if isinstance(features, (list, tuple)):
                    # 使用最后一层特征
                    out_channels = features[-1].shape[1]
                    # 创建特征提取器，返回最后一层
                    return_nodes = {f'blocks.{len(features)-1}': '0'}
                else:
                    out_channels = features.shape[1]
                    return_nodes = {'blocks': '0'}
            
            # 由于 timm 的 features_only 已经返回特征，我们需要包装它
            class EfficientNetLite0Backbone(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model
                    self.out_channels = out_channels
                
                def forward(self, x):
                    features = self.model(x)
                    if isinstance(features, (list, tuple)):
                        return {'0': features[-1]}
                    else:
                        return {'0': features}
            
            backbone = EfficientNetLite0Backbone(model)
            return backbone
            
        except Exception as e:
            print(f"使用 timm 创建模型失败: {e}")
            # 备用方案：使用 torchvision 的 EfficientNet-B0 作为近似
            print("使用 torchvision 的 EfficientNet-B0 作为替代")
            model = models.efficientnet_b0(pretrained=True)
            return_nodes = {'features': '0'}
            backbone = create_feature_extractor(model, return_nodes=return_nodes)
            backbone.out_channels = 1280
            return backbone
    
    # 如果 USE_LITE_BUILDER 为 True 且成功创建，这里可以添加相应的逻辑
    # 但由于 efficientnet_lite_builder.py 是 TensorFlow 版本，这里不会执行
    # 如果到达这里，使用 torchvision 作为最终备用方案
    print("使用 torchvision 的 EfficientNet-B0 作为最终备用方案")
    model = models.efficientnet_b0(pretrained=True)
    return_nodes = {'features': '0'}
    backbone = create_feature_extractor(model, return_nodes=return_nodes)
    backbone.out_channels = 1280
    return backbone


def create_model(num_classes, model_name="EfficientDet-Lite0"):
    """
    创建基于 EfficientNet-Lite0 的 Faster R-CNN 模型
    """
    # 创建 EfficientNet-Lite0 backbone
    backbone = create_efficientnet_lite0_backbone()
    
    # 配置 anchor generator
    # EfficientNet-Lite0 通常使用较小的 anchor sizes
    anchor_generator = AnchorsGenerator(sizes=((14),), aspect_ratios=((1.0),))
    
    # 配置 ROI Pooler
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], 
                                                     output_size=[7, 7], 
                                                     sampling_ratio=2)
    
    # 创建 Faster R-CNN 模型
    model = FasterRCNN(backbone=backbone,
                      num_classes=num_classes,
                      rpn_anchor_generator=anchor_generator,
                      box_roi_pool=roi_pooler)
    
    return model


def train_model(model_name="EfficientDet-Lite0"):
    """
    训练 EfficientDet-Lite0 模型
    使用与 train_desnet.py 相同的训练数据和策略
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training {model_name} using {device.type} device.")
    
    # 清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU memory before training: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        print(f"GPU memory cached: {torch.cuda.memory_reserved()/1024**3:.2f} GB")

    # 用来保存coco_info的文件
    results_file = f"results_{model_name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.txt"

    # 检查保存权重文件夹是否存在，不存在则创建
    save_dir = f"save_weights_{model_name}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    data_transform = {
        "train": transforms.Compose([transforms.ToTensor()]),
        "val": transforms.Compose([transforms.ToTensor()])
    }

    VOC_root = "./"  # VOCdevkit                             
    aspect_ratio_group_factor = 1
    batch_size = 1
    amp = True

    if os.path.exists(os.path.join(VOC_root, "VOCdevkit")) is False:
        raise FileNotFoundError("VOCdevkit dose not in path:'{}'.".format(VOC_root))

    train_dataset = VOCDataSet(VOC_root, "2012", data_transform["train"], "train.txt")
    train_sampler = None

    if aspect_ratio_group_factor >= 0:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        group_ids = create_aspect_ratio_groups(train_dataset, k=aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, batch_size)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using %g dataloader workers' % nw)

    if train_sampler:
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                      batch_sampler=train_batch_sampler,
                                                      pin_memory=False,
                                                      num_workers=1,
                                                      collate_fn=train_dataset.collate_fn)
    else:
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                      batch_size=batch_size,
                                                      shuffle=True,
                                                      pin_memory=False,
                                                      num_workers=1,
                                                      collate_fn=train_dataset.collate_fn)

    val_dataset = VOCDataSet(VOC_root, "2012", data_transform["val"], "val.txt")
    val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                                batch_size=1,
                                                shuffle=False,
                                                pin_memory=False,
                                                num_workers=1,
                                                collate_fn=val_dataset.collate_fn)

    model = create_model(num_classes=2, model_name=model_name)
    print(model)

    model.to(device)
    scaler = torch.cuda.amp.GradScaler() if amp else None

    train_loss = []
    learning_rate = []
    val_map = []

    # 第一阶段：冻结backbone
    for param in model.backbone.parameters():
        param.requires_grad = False

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001,
                              momentum=0.9, weight_decay=0.0002)

    init_epochs = 5
    num_epochs = 50
    for epoch in range(init_epochs):
        mean_loss, lr = utils.train_one_epoch(model, optimizer, train_data_loader,
                                            device, epoch, num_epochs=num_epochs, print_freq=50,
                                            warmup=True, scaler=scaler)
        train_loss.append(mean_loss.item())
        learning_rate.append(lr)

        coco_info = utils.evaluate(model, val_data_loader, device=device, epoch=epoch, num_epochs=0)
        val_map.append(coco_info[1])
        
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"Epoch {epoch} - GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    # 第二阶段：解冻部分backbone
    # 对于 EfficientNet-Lite0，我们需要根据实际的层结构来解冻
    # 这里我们解冻后几层
    backbone_params = list(model.backbone.named_parameters())
    # 冻结前70%的参数，解冻后30%
    freeze_ratio = 0.7
    freeze_idx = int(len(backbone_params) * freeze_ratio)
    
    for idx, (name, parameter) in enumerate(backbone_params):
        if idx < freeze_idx:
            parameter.requires_grad = False
        else:
            parameter.requires_grad = True

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001,
                              momentum=0.9, weight_decay=0.00005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                 step_size=2000,
                                                 gamma=0.1)

    for epoch in range(init_epochs, num_epochs+init_epochs, 1):
        mean_loss, lr = utils.train_one_epoch(model, optimizer, train_data_loader,
                                            device, epoch, num_epochs=num_epochs, print_freq=50,
                                            warmup=True, scaler=scaler)
        train_loss.append(mean_loss.item())
        learning_rate.append(lr)

        lr_scheduler.step()

        coco_info = utils.evaluate(model, val_data_loader, device=device, epoch=epoch, num_epochs=num_epochs)
        val_map.append(coco_info[1])
        
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"Epoch {epoch} - GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

        if epoch in range(num_epochs+init_epochs)[-4:]:
            save_files = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch}
            torch.save(save_files, f"{save_dir}/{model_name}-model-{epoch}.pth")

    # 绘制损失和学习率曲线
    # if len(train_loss) != 0 and len(learning_rate) != 0:
    #     from plot_curve import plot_loss_and_lr
    #     plot_loss_and_lr(train_loss, learning_rate, save_path=f"{save_dir}/loss_lr_curve.png")
    #
    # # 绘制mAP曲线
    # if len(val_map) != 0:
    #     from plot_curve import plot_map
    #     plot_map(val_map, save_path=f"{save_dir}/map_curve.png")


def main():
    """
    主函数
    """
    model_name = "EfficientDet-Lite0"
    print(f"\n开始训练模型: {model_name}")
    print("=" * 50)
    train_model(model_name)
    print("=" * 50)
    print(f"完成模型 {model_name} 的训练\n")


if __name__ == "__main__":
    main()

