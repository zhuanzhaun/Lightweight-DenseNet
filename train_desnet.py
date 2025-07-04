import os
import datetime
import sys
import torch
import torchvision
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor
import transforms
from network_files import FasterRCNN, AnchorsGenerator
from backbone import densenet_literature17,densenet_literature20 ,densenet_mod1 , densenet_mod2, densenet_mod3,densenet_mod4 ,densenet_mod5, densenet_mod6
from backbone import densenet121yuan,densenet169
from my_dataset import VOCDataSet
from train_utils import GroupedBatchSampler, create_aspect_ratio_groups
from train_utils import train_eval_utils as utils
sys.setrecursionlimit(100000000)
# /root/miniconda3/envs/myconda/bin/python

def create_model(num_classes, model_name):
    if model_name == "densenet_literature17":
        model = densenet_literature17()
        model.features.conv0 = torch.nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1, bias=False)
        # 先获取实际的输出通道数
        with torch.no_grad():
            test_input = torch.randn(1, 3, 608, 608)
            test_output = model.features(test_input)
            actual_channels = test_output.shape[1]
        # 添加1x1卷积层，将DenseNet输出通道数降维到24
        model.features.add_module("final_conv", torch.nn.Conv2d(actual_channels, 128, kernel_size=1, stride=1, bias=False))
        return_nodes = {'features.final_conv': '0'}
    elif model_name == "densenet_literature20":
        model = densenet_literature20()
        model.features.conv0 = torch.nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1, bias=False)
        # 先获取实际的输出通道数
        with torch.no_grad():
            test_input = torch.randn(1, 3, 608, 608)
            test_output = model.features(test_input)
            actual_channels = test_output.shape[1]
        # 添加1x1卷积层，将DenseNet输出通道数降维到24
        model.features.add_module("final_conv", torch.nn.Conv2d(actual_channels, 128, kernel_size=1, stride=1, bias=False))
        return_nodes = {'features.final_conv': '0'}
    elif model_name == "densenet_mod1":
        model = densenet_mod1()
        model.features.conv0 = torch.nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1, bias=False)
        # 先获取实际的输出通道数
        with torch.no_grad():
            test_input = torch.randn(1, 3, 608, 608)
            test_output = model.features(test_input)
            actual_channels = test_output.shape[1]
        # 添加1x1卷积层，将DenseNet输出通道数降维到24
        model.features.add_module("final_conv", torch.nn.Conv2d(actual_channels, 128, kernel_size=1, stride=1, bias=False))
        return_nodes = {'features.final_conv': '0'}
    elif model_name == "densenet_mod2":
        model = densenet_mod2()
        model.features.conv0 = torch.nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1, bias=False)
        # 先获取实际的输出通道数
        with torch.no_grad():
            test_input = torch.randn(1, 3, 608, 608)
            test_output = model.features(test_input)
            actual_channels = test_output.shape[1]
        # 添加1x1卷积层，将DenseNet输出通道数降维到24
        model.features.add_module("final_conv", torch.nn.Conv2d(actual_channels, 128, kernel_size=1, stride=1, bias=False))
        return_nodes = {'features.final_conv': '0'}
    elif model_name == "densenet_mod3":
        model = densenet_mod3()
        model.features.conv0 = torch.nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1, bias=False)
        # 先获取实际的输出通道数
        with torch.no_grad():
            test_input = torch.randn(1, 3, 608, 608)
            test_output = model.features(test_input)
            actual_channels = test_output.shape[1]
        # 添加1x1卷积层，将DenseNet输出通道数降维到24
        model.features.add_module("final_conv", torch.nn.Conv2d(actual_channels, 128, kernel_size=1, stride=1, bias=False))
        return_nodes = {'features.final_conv': '0'}
    elif model_name == "densenet_mod4":
        model = densenet_mod4()
        model.features.conv0 = torch.nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1, bias=False)
        # 先获取实际的输出通道数
        with torch.no_grad():
            test_input = torch.randn(1, 3, 608, 608)
            test_output = model.features(test_input)
            actual_channels = test_output.shape[1]
        # 添加1x1卷积层，将DenseNet输出通道数降维到24
        model.features.add_module("final_conv", torch.nn.Conv2d(actual_channels, 128, kernel_size=1, stride=1, bias=False))
        return_nodes = {'features.final_conv': '0'}
    elif model_name == "densenet_mod5":
        model = densenet_mod5()
        model.features.conv0 = torch.nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1, bias=False)
        # 先获取实际的输出通道数
        with torch.no_grad():
            test_input = torch.randn(1, 3, 608, 608)
            test_output = model.features(test_input)
            actual_channels = test_output.shape[1]
        # 添加1x1卷积层，将DenseNet输出通道数降维到24
        model.features.add_module("final_conv", torch.nn.Conv2d(actual_channels, 128, kernel_size=1, stride=1, bias=False))
        return_nodes = {'features.final_conv': '0'}
    elif model_name == "densenet_mod6":
        model = densenet_mod6()
        model.features.conv0 = torch.nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1, bias=False)
        # 先获取实际的输出通道数
        with torch.no_grad():
            test_input = torch.randn(1, 3, 608, 608)
            test_output = model.features(test_input)
            actual_channels = test_output.shape[1]
        # 添加1x1卷积层，将DenseNet输出通道数降维到24
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
        # 去掉全连接层和池化层，只保留特征提取部分
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
                      # rpn_pre_nms_top_n_train=200,
                      # rpn_post_nms_top_n_train=200,
                      # rpn_batch_size_per_image=128,
                      # rpn_pre_nms_top_n_test=200,
                      # rpn_post_nms_top_n_test=200

    return model

def train_model(model_name):
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
    model_names = [
         # "densenet121yuan",
          "densenet169"
        # "resnet50",
        # "efficientnet_b0"
        # "shufflenetv2"
        #"densenet_literature17"
        # "densenet_literature20"
        #"densenet_mod5"

    ]
    
    for model_name in model_names:
        print(f"\n开始训练模型: {model_name}")
        train_model(model_name)
        print(f"完成模型 {model_name} 的训练\n")

if __name__ == "__main__":
    main()
