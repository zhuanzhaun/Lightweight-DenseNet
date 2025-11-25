import os
import datetime
import sys
import torch
import yaml
from pathlib import Path
import transforms
from my_dataset import VOCDataSet
sys.setrecursionlimit(100000000)

# 尝试导入 NanoDet-Plus
try:
    from nanodet.util import cfg, load_config, Logger
    from nanodet.trainer import Trainer
    from nanodet.data.collate import collate_function
    from nanodet.data.dataset import build_dataset
    NANODET_AVAILABLE = True
    print("成功导入 NanoDet-Plus")
except ImportError:
    NANODET_AVAILABLE = False
    print("警告: 未找到 NanoDet-Plus 库")
    print("请安装 NanoDet-Plus: pip install nanodet")
    print("或从 GitHub 克隆: git clone https://github.com/RangiLyu/nanodet.git")
    raise


def create_nanodet_config(voc_root, year="2012", save_dir="save_weights_NanoDet-Plus"):
    """
    创建 NanoDet-Plus 配置文件
    使用与 train_desnet.py 相同的数据路径
    """
    # 读取类别信息
    json_file = './pascal_crack.json'
    assert os.path.exists(json_file), "{} file not exist.".format(json_file)
    import json
    with open(json_file, 'r') as f:
        class_dict = json.load(f)
    
    class_names = list(class_dict.keys())
    num_classes = len(class_names)
    
    # 构建数据集路径
    voc_dataset_path = os.path.join(voc_root, "VOCdevkit", f"VOC{year}")
    train_txt_path = os.path.join(voc_dataset_path, "ImageSets", "Main", "train.txt")
    val_txt_path = os.path.join(voc_dataset_path, "ImageSets", "Main", "val.txt")
    
    # 创建 NanoDet-Plus 配置文件
    config = {
        'model': {
            'arch': {
                'backbone': {
                    'name': 'ShuffleNetV2',
                    'model_size': '1.0x',
                    'out_stages': [2, 3, 4],
                    'activation': 'relu'
                },
                'fpn': {
                    'name': 'PAN',
                    'in_channels': [116, 232, 464],
                    'out_channels': 96,
                    'start_level': 0,
                    'num_outs': 3,
                    'activation': 'relu'
                },
                'head': {
                    'name': 'NanoDetPlusHead',
                    'num_classes': num_classes,
                    'input_channel': 96,
                    'feat_channels': 96,
                    'stacked_convs': 2,
                    'share_cls_reg': True,
                    'octave_base_scale': 5,
                    'scales_per_octave': 1,
                    'strides': [8, 16, 32],
                    'reg_max': 7,
                    'norm_cfg': {'type': 'BN'},
                    'activation': 'relu',
                    'loss': {
                        'loss_qfl': {
                            'name': 'QualityFocalLoss',
                            'use_sigmoid': True,
                            'beta': 2.0,
                            'loss_weight': 1.0
                        },
                        'loss_bbox': {
                            'name': 'GIoULoss',
                            'loss_weight': 2.0
                        },
                        'loss_dfl': {
                            'name': 'DistributionFocalLoss',
                            'loss_weight': 0.25
                        }
                    }
                }
            },
            'pretrained': True,
            'backbone': {
                'pretrained': True
            }
        },
        'data': {
            'train': {
                'name': 'VOCDataset',
                'img_path': os.path.join(voc_dataset_path, "JPEGImages"),
                'ann_path': os.path.join(voc_dataset_path, "Annotations"),
                'input_size': [416, 416],  # 可以调整为 608 以匹配 train_desnet.py
                'keep_ratio': True,
                'augment': {
                    'flip': 0.5,
                    'hsv_h': 0.015,
                    'hsv_s': 0.7,
                    'hsv_v': 0.4,
                    'degrees': 10,
                    'translate': 0.1,
                    'scale': 0.5,
                    'shear': 2,
                    'perspective': 0.0,
                    'flipud': 0.0,
                    'fliplr': 0.5,
                    'mosaic': 1.0,
                    'mixup': 0.0
                },
                'data_set': train_txt_path
            },
            'val': {
                'name': 'VOCDataset',
                'img_path': os.path.join(voc_dataset_path, "JPEGImages"),
                'ann_path': os.path.join(voc_dataset_path, "Annotations"),
                'input_size': [416, 416],
                'keep_ratio': True,
                'augment': None,
                'data_set': val_txt_path
            },
            'num_classes': num_classes,
            'class_names': class_names
        },
        'device': {
            'gpu_ids': [0] if torch.cuda.is_available() else [],
            'batch_per_gpu': 1,  # 与 train_desnet.py 一致
            'num_workers': 1,  # 与 train_desnet.py 一致
            'pin_memory': False
        },
        'schedule': {
            'total_epochs': 50,  # 与 train_desnet.py 一致
            'lr_schedule': {
                'type': 'cosine',
                'warmup': 'linear',
                'warmup_epochs': 5,
                'warmup_iters': 200,
                'warmup_lr': 0.0001
            },
            'optimizer': {
                'type': 'SGD',
                'lr': 0.001,  # 与 train_desnet.py 一致
                'weight_decay': 0.0002,  # 第一阶段
                'momentum': 0.9  # 与 train_desnet.py 一致
            }
        },
        'save_dir': save_dir,
        'resume': {
            'resume_path': None,
            'load_model': None,
            'load_optimizer': None
        },
        'evaluate': {
            'interval': 1,
            'metric': 'mAP'
        }
    }
    
    return config


def train_nanodet_plus():
    """
    训练 NanoDet-Plus 模型
    使用与 train_desnet.py 相同的训练数据和策略
    """
    if not NANODET_AVAILABLE:
        raise ImportError("NanoDet-Plus 库未安装，请先安装")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training NanoDet-Plus using {device.type} device.")
    
    # 清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU memory before training: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        print(f"GPU memory cached: {torch.cuda.memory_reserved()/1024**3:.2f} GB")

    # 用来保存训练结果的文件
    results_file = f"results_NanoDet-Plus_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.txt"

    # 检查保存权重文件夹是否存在，不存在则创建
    save_dir = f"save_weights_NanoDet-Plus"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 数据路径配置（与train_desnet.py保持一致）
    VOC_root = "./"  # VOCdevkit
    year = "2012"
    
    if os.path.exists(os.path.join(VOC_root, "VOCdevkit")) is False:
        raise FileNotFoundError("VOCdevkit dose not in path:'{}'.".format(VOC_root))

    # 创建配置文件
    config = create_nanodet_config(VOC_root, year, save_dir)
    
    # 保存配置文件
    config_path = os.path.join(save_dir, "nanodet_plus_config.yml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"配置文件已保存: {config_path}")

    # 加载配置到 NanoDet 的 cfg
    load_config(cfg, config_path)
    
    # 设置日志
    logger = Logger(local_rank=0, save_dir=cfg.save_dir, use_tensorboard=True)
    
    # 初始化训练器
    trainer = Trainer(cfg, logger)
    
    # 第一阶段：冻结backbone（前5个epoch）
    print("\n开始第一阶段训练（冻结backbone，5个epoch）...")
    
    # 冻结backbone参数
    try:
        if hasattr(trainer.model, 'backbone'):
            for param in trainer.model.backbone.parameters():
                param.requires_grad = False
            print("已冻结backbone参数")
        elif hasattr(trainer.model, 'module') and hasattr(trainer.model.module, 'backbone'):
            # 处理 DataParallel 包装的情况
            for param in trainer.model.module.backbone.parameters():
                param.requires_grad = False
            print("已冻结backbone参数（DataParallel模式）")
    except Exception as e:
        print(f"冻结backbone时出现警告: {e}")
    
    # 修改优化器配置（第一阶段）
    try:
        cfg.schedule.optimizer.weight_decay = 0.0002
        # 重新创建优化器（如果可能）
        if hasattr(trainer, 'optimizer'):
            trainer.optimizer = torch.optim.SGD(
                [p for p in trainer.model.parameters() if p.requires_grad],
                lr=cfg.schedule.optimizer.lr,
                momentum=cfg.schedule.optimizer.momentum,
                weight_decay=cfg.schedule.optimizer.weight_decay
            )
    except Exception as e:
        print(f"修改优化器时出现警告: {e}")
    
    # 第一阶段训练（5个epoch）
    init_epochs = 5
    try:
        for epoch in range(init_epochs):
            trainer.train_epoch(epoch)
            trainer.evaluate(epoch)
            
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print(f"Epoch {epoch} - GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    except Exception as e:
        print(f"第一阶段训练时出现错误: {e}")
        print("继续执行第二阶段训练...")
    
    # 第二阶段：解冻部分backbone（后45个epoch）
    print("\n开始第二阶段训练（解冻部分backbone，45个epoch）...")
    
    # 解冻部分backbone
    try:
        if hasattr(trainer.model, 'backbone'):
            backbone_params = list(trainer.model.backbone.named_parameters())
            freeze_ratio = 0.7  # 冻结前70%，解冻后30%
            freeze_idx = int(len(backbone_params) * freeze_ratio)
            
            for idx, (name, param) in enumerate(backbone_params):
                if idx < freeze_idx:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            print(f"已解冻backbone后{100-int(freeze_ratio*100)}%的参数")
        elif hasattr(trainer.model, 'module') and hasattr(trainer.model.module, 'backbone'):
            backbone_params = list(trainer.model.module.backbone.named_parameters())
            freeze_ratio = 0.7
            freeze_idx = int(len(backbone_params) * freeze_ratio)
            
            for idx, (name, param) in enumerate(backbone_params):
                if idx < freeze_idx:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            print(f"已解冻backbone后{100-int(freeze_ratio*100)}%的参数（DataParallel模式）")
    except Exception as e:
        print(f"解冻backbone时出现警告: {e}")
    
    # 修改优化器配置（第二阶段）
    try:
        cfg.schedule.optimizer.weight_decay = 0.00005
        if hasattr(trainer, 'optimizer'):
            trainer.optimizer = torch.optim.SGD(
                [p for p in trainer.model.parameters() if p.requires_grad],
                lr=cfg.schedule.optimizer.lr,
                momentum=cfg.schedule.optimizer.momentum,
                weight_decay=cfg.schedule.optimizer.weight_decay
            )
    except Exception as e:
        print(f"修改优化器时出现警告: {e}")
    
    # 第二阶段训练（45个epoch）
    num_epochs = 50
    try:
        for epoch in range(init_epochs, num_epochs):
            trainer.train_epoch(epoch)
            trainer.evaluate(epoch)
            
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print(f"Epoch {epoch} - GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
            
            # 保存最后4个epoch的模型
            if epoch >= num_epochs - 4:
                checkpoint_path = os.path.join(save_dir, f"NanoDet-Plus-model-{epoch}.pth")
                try:
                    model_state = trainer.model.state_dict() if not hasattr(trainer.model, 'module') else trainer.model.module.state_dict()
                    optimizer_state = trainer.optimizer.state_dict() if hasattr(trainer, 'optimizer') else None
                    checkpoint = {
                        'model': model_state,
                        'epoch': epoch
                    }
                    if optimizer_state:
                        checkpoint['optimizer'] = optimizer_state
                    torch.save(checkpoint, checkpoint_path)
                    print(f"模型已保存: {checkpoint_path}")
                except Exception as e:
                    print(f"保存模型时出现错误: {e}")
    except Exception as e:
        print(f"第二阶段训练时出现错误: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n训练完成！模型保存在: {save_dir}")
    print(f"训练结果文件: {results_file}")


def main():
    """
    主函数
    """
    print("=" * 50)
    print("开始训练 NanoDet-Plus 模型")
    print("使用与 train_desnet.py 相同的训练数据")
    print("=" * 50)
    
    try:
        train_nanodet_plus()
        print("\n" + "=" * 50)
        print("训练成功完成！")
        print("=" * 50)
    except Exception as e:
        print(f"\n训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

