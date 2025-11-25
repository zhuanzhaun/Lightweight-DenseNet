import os
import datetime
import sys
import torch
import torchvision
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor
import torch.nn as nn
import torch.nn.functional as F
import transforms
from network_files import FasterRCNN, AnchorsGenerator
from backbone import densenet_literature17, densenet_literature20, densenet_mod1, densenet_mod2, densenet_mod3, densenet_mod4, densenet_mod5, densenet_mod6
from backbone import densenet121yuan, densenet169
from my_dataset import VOCDataSet
from train_utils import GroupedBatchSampler, create_aspect_ratio_groups
from train_utils import train_eval_utils as utils

# 导入 DRConv
sys.path.append(os.path.join(os.path.dirname(__file__), '自适应感受野对比实验'))
try:
    from DRConv import DRConv
except ImportError:
    print("警告: 无法导入 DRConv，请确保 DRConv.py 在正确路径")
    raise

sys.setrecursionlimit(100000000)


class DRConvWrapper(nn.Module):
    """
    DRConv 包装器，自动生成 mask 和 Alpha 参数
    这样可以将 DRConv 作为标准卷积的替代品使用
    """
    def __init__(self, drconv_module):
        super(DRConvWrapper, self).__init__()
        self.drconv = drconv_module
    
    def forward(self, x):
        """
        自动生成 mask 和 Alpha 参数
        mask: 全1掩码，表示所有区域都关注
        Alpha: 初始化为1，可以后续学习
        """
        batch_size, channels, height, width = x.shape
        
        # 创建全1的mask（表示所有区域都关注）
        mask = torch.ones(batch_size, height, width, device=x.device, dtype=x.dtype)
        
        # 创建 Alpha（初始化为1，可以后续学习）
        alpha = torch.ones(batch_size, 1, height, width, device=x.device, dtype=x.dtype)
        
        return self.drconv(x, mask, alpha, use_alpha=False, beta=1.0)


def replace_conv_with_drconv(module, replace_1x1=False, groups_num=8, num_W=8):
    """
    递归地将模块中的 Conv2d 层替换为 DRConv
    
    Args:
        module: 要处理的模块
        replace_1x1: 是否替换 1x1 卷积（默认 False，因为 1x1 卷积通常不需要自适应感受野）
        groups_num: DRConv 的分组数
        num_W: DRConv 的权重模板数
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d):
            # 判断是否应该替换
            kernel_size = child.kernel_size
            is_1x1 = kernel_size == (1, 1) or (isinstance(kernel_size, tuple) and kernel_size[0] == 1 and kernel_size[1] == 1)
            
            if not is_1x1 or replace_1x1:
                # 创建 DRConv
                drconv = DRConv(
                    in_channels=child.in_channels,
                    out_channels=child.out_channels,
                    kernel_size=kernel_size[0] if isinstance(kernel_size, tuple) else kernel_size,
                    stride=child.stride[0] if isinstance(child.stride, tuple) else child.stride,
                    padding=child.padding[0] if isinstance(child.padding, tuple) else child.padding,
                    groups=child.groups,
                    bias=child.bias is not None,
                    dilation=child.dilation[0] if isinstance(child.dilation, tuple) else child.dilation,
                    groups_num=groups_num,
                    num_W=num_W
                )
                
                # 如果原卷积有偏置，尝试复制（DRConv 的偏置初始化可能不同）
                if child.bias is not None and drconv.bias is not None:
                    with torch.no_grad():
                        drconv.bias.data.copy_(child.bias.data)
                
                # 用包装器包装 DRConv
                wrapped_drconv = DRConvWrapper(drconv)
                setattr(module, name, wrapped_drconv)
                print(f"  替换 {name}: Conv2d -> DRConv (kernel_size={kernel_size}, in={child.in_channels}, out={child.out_channels})")
        else:
            # 递归处理子模块
            replace_conv_with_drconv(child, replace_1x1, groups_num, num_W)


def create_model(num_classes, model_name, use_drconv=True, drconv_groups_num=8, drconv_num_W=8, replace_1x1_conv=False):
    """
    创建模型，可选择使用 DRConv 替换标准卷积
    
    Args:
        num_classes: 类别数
        model_name: 模型名称
        use_drconv: 是否使用 DRConv 替换卷积层
        drconv_groups_num: DRConv 的分组数
        drconv_num_W: DRConv 的权重模板数
        replace_1x1_conv: 是否替换 1x1 卷积
    """
    if model_name == "densenet_literature17":
        model = densenet_literature17()
        model.features.conv0 = torch.nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1, bias=False)
        with torch.no_grad():
            test_input = torch.randn(1, 3, 608, 608)
            test_output = model.features(test_input)
            actual_channels = test_output.shape[1]
        model.features.add_module("final_conv", torch.nn.Conv2d(actual_channels, 128, kernel_size=1, stride=1, bias=False))
        return_nodes = {'features.final_conv': '0'}
    elif model_name == "densenet_literature20":
        model = densenet_literature20()
        model.features.conv0 = torch.nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1, bias=False)
        with torch.no_grad():
            test_input = torch.randn(1, 3, 608, 608)
            test_output = model.features(test_input)
            actual_channels = test_output.shape[1]
        model.features.add_module("final_conv", torch.nn.Conv2d(actual_channels, 128, kernel_size=1, stride=1, bias=False))
        return_nodes = {'features.final_conv': '0'}
    elif model_name == "densenet_mod1":
        model = densenet_mod1()
        model.features.conv0 = torch.nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1, bias=False)
        with torch.no_grad():
            test_input = torch.randn(1, 3, 608, 608)
            test_output = model.features(test_input)
            actual_channels = test_output.shape[1]
        model.features.add_module("final_conv", torch.nn.Conv2d(actual_channels, 128, kernel_size=1, stride=1, bias=False))
        return_nodes = {'features.final_conv': '0'}
    elif model_name == "densenet_mod2":
        model = densenet_mod2()
        model.features.conv0 = torch.nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1, bias=False)
        with torch.no_grad():
            test_input = torch.randn(1, 3, 608, 608)
            test_output = model.features(test_input)
            actual_channels = test_output.shape[1]
        model.features.add_module("final_conv", torch.nn.Conv2d(actual_channels, 128, kernel_size=1, stride=1, bias=False))
        return_nodes = {'features.final_conv': '0'}
    elif model_name == "densenet_mod3":
        model = densenet_mod3()
        model.features.conv0 = torch.nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1, bias=False)
        with torch.no_grad():
            test_input = torch.randn(1, 3, 608, 608)
            test_output = model.features(test_input)
            actual_channels = test_output.shape[1]
        model.features.add_module("final_conv", torch.nn.Conv2d(actual_channels, 128, kernel_size=1, stride=1, bias=False))
        return_nodes = {'features.final_conv': '0'}
    elif model_name == "densenet_mod4":
        model = densenet_mod4()
        model.features.conv0 = torch.nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1, bias=False)
        with torch.no_grad():
            test_input = torch.randn(1, 3, 608, 608)
            test_output = model.features(test_input)
            actual_channels = test_output.shape[1]
        model.features.add_module("final_conv", torch.nn.Conv2d(actual_channels, 128, kernel_size=1, stride=1, bias=False))
        return_nodes = {'features.final_conv': '0'}
    elif model_name == "densenet_mod5":
        model = densenet_mod5()
        model.features.conv0 = torch.nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1, bias=False)
        with torch.no_grad():
            test_input = torch.randn(1, 3, 608, 608)
            test_output = model.features(test_input)
            actual_channels = test_output.shape[1]
        model.features.add_module("final_conv", torch.nn.Conv2d(actual_channels, 128, kernel_size=1, stride=1, bias=False))
        return_nodes = {'features.final_conv': '0'}
    elif model_name == "densenet_mod6":
        model = densenet_mod6()
        model.features.conv0 = torch.nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1, bias=False)
        with torch.no_grad():
            test_input = torch.randn(1, 3, 608, 608)
            test_output = model.features(test_input)
            actual_channels = test_output.shape[1]
        model.features.add_module("final_conv", torch.nn.Conv2d(actual_channels, 128, kernel_size=1, stride=1, bias=False))
        return_nodes = {'features.final_conv': '0'}
    elif model_name == "densenet121yuan":
        model = densenet121yuan()
        model.features.conv0 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        with torch.no_grad():
            test_input = torch.randn(1, 3, 608, 608)
            test_output = model.features(test_input)
            actual_channels = test_output.shape[1]
        model.features.add_module("final_conv", torch.nn.Conv2d(actual_channels, 128, kernel_size=1, stride=1, bias=False))
        return_nodes = {'features.final_conv': '0'}
    elif model_name == "densenet169":
        model = densenet169()
        model.features.conv0 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        with torch.no_grad():
            test_input = torch.randn(1, 3, 608, 608)
            test_output = model.features(test_input)
            actual_channels = test_output.shape[1]
        model.features.add_module("final_conv",
                                  torch.nn.Conv2d(actual_channels, 128, kernel_size=1, stride=1, bias=False))
        return_nodes = {'features.final_conv': '0'}
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=True)
        return_nodes = {'layer4': '0'}
        backbone = create_feature_extractor(model, return_nodes=return_nodes)
        backbone.out_channels = 2048
        anchor_generator = AnchorsGenerator(sizes=((14),), aspect_ratios=((1.0),))
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=[14, 14], sampling_ratio=2)
        model = FasterRCNN(backbone=backbone,
                          num_classes=num_classes,
                          rpn_anchor_generator=anchor_generator,
                          box_roi_pool=roi_pooler)
        return model
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=True)
        return_nodes = {'features': '0'}
        backbone = create_feature_extractor(model, return_nodes=return_nodes)
        backbone.out_channels = 1280
        anchor_generator = AnchorsGenerator(sizes=((14),), aspect_ratios=((1.0),))
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=[14, 14], sampling_ratio=2)
        model = FasterRCNN(backbone=backbone,
                          num_classes=num_classes,
                          rpn_anchor_generator=anchor_generator,
                          box_roi_pool=roi_pooler)
        return model
    elif model_name == "shufflenetv2":
        model = models.shufflenet_v2_x1_0(pretrained=True)
        return_nodes = {'conv5': '0'}
        backbone = create_feature_extractor(model, return_nodes=return_nodes)
        backbone.out_channels = 1024
        anchor_generator = AnchorsGenerator(sizes=((14),), aspect_ratios=((1.0),))
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=[7, 7], sampling_ratio=2)
        model = FasterRCNN(backbone=backbone,
                          num_classes=num_classes,
                          rpn_anchor_generator=anchor_generator,
                          box_roi_pool=roi_pooler)
        return model
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    # 如果使用 DRConv，替换卷积层
    if use_drconv:
        print(f"\n使用 DRConv 替换 {model_name} 中的卷积层...")
        print(f"DRConv 参数: groups_num={drconv_groups_num}, num_W={drconv_num_W}, replace_1x1={replace_1x1_conv}")
        
        # 替换 features 中的卷积层（不包括 conv0 和 final_conv，它们会在后面处理）
        # 先保存 conv0 和 final_conv
        original_conv0 = model.features.conv0
        original_final_conv = model.features.final_conv if hasattr(model.features, 'final_conv') else None
        
        # 替换 features 中的其他卷积层
        replace_conv_with_drconv(model.features, replace_1x1=replace_1x1_conv, 
                                groups_num=drconv_groups_num, num_W=drconv_num_W)
        
        # 恢复 conv0（第一层通常保持标准卷积）
        model.features.conv0 = original_conv0
        
        # 可选：替换 final_conv 为 DRConv
        if replace_1x1_conv and original_final_conv is not None:
            drconv_final = DRConv(
                in_channels=original_final_conv.in_channels,
                out_channels=original_final_conv.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
                bias=original_final_conv.bias is not None,
                dilation=1,
                groups_num=drconv_groups_num,
                num_W=drconv_num_W
            )
            model.features.final_conv = DRConvWrapper(drconv_final)
            print(f"  替换 final_conv: Conv2d -> DRConv")
        elif original_final_conv is not None:
            model.features.final_conv = original_final_conv
        
        print(f"DRConv 替换完成！\n")
    
    backbone = create_feature_extractor(model, return_nodes=return_nodes)
    backbone.out_channels = 128
    anchor_generator = AnchorsGenerator(sizes=((14),),
                                      aspect_ratios=((1.0),))

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                  output_size=[7, 7],
                                                  sampling_ratio=2)

    model = FasterRCNN(backbone=backbone,
                      num_classes=num_classes,
                      rpn_anchor_generator=anchor_generator,
                      box_roi_pool=roi_pooler)

    return model


def train_model(model_name, use_drconv=True, drconv_groups_num=8, drconv_num_W=8, replace_1x1_conv=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    drconv_str = "with DRConv" if use_drconv else "without DRConv"
    print(f"Training {model_name} {drconv_str} using {device.type} device.")
    
    # 清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU memory before training: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        print(f"GPU memory cached: {torch.cuda.memory_reserved()/1024**3:.2f} GB")

    # 用来保存coco_info的文件
    suffix = "_drconv" if use_drconv else ""
    results_file = f"results_{model_name}{suffix}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.txt"

    # 检查保存权重文件夹是否存在，不存在则创建
    save_dir = f"save_weights_{model_name}{suffix}"
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

    model = create_model(num_classes=2, model_name=model_name, 
                        use_drconv=use_drconv, 
                        drconv_groups_num=drconv_groups_num,
                        drconv_num_W=drconv_num_W,
                        replace_1x1_conv=replace_1x1_conv)
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
    for name, parameter in model.backbone.named_parameters():
        split_name = name.split(".")[0]
        if split_name in ["0", "1", "2", "3"]:
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
            torch.save(save_files, f"{save_dir}/{model_name}{suffix}-model-{epoch}.pth")


def main():
    model_names = [
        "densenet_mod6",  # 可以修改为其他模型
        # "densenet169",
        # "densenet121yuan",
        # "densenet_mod5",
        # "densenet_literature17",
        # "densenet_literature20",
    ]
    
    # DRConv 配置
    use_drconv = True  # 是否使用 DRConv
    drconv_groups_num = 8  # DRConv 分组数
    drconv_num_W = 8  # DRConv 权重模板数
    replace_1x1_conv = False  # 是否替换 1x1 卷积（通常不需要）
    
    for model_name in model_names:
        print(f"\n{'='*60}")
        print(f"开始训练模型: {model_name}")
        if use_drconv:
            print(f"使用 DRConv 实现自适应感受野")
            print(f"DRConv 参数: groups_num={drconv_groups_num}, num_W={drconv_num_W}")
        print(f"{'='*60}\n")
        
        train_model(model_name, 
                   use_drconv=use_drconv,
                   drconv_groups_num=drconv_groups_num,
                   drconv_num_W=drconv_num_W,
                   replace_1x1_conv=replace_1x1_conv)
        print(f"\n完成模型 {model_name} 的训练\n")


if __name__ == "__main__":
    main()

