import math
import sys
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch

from .coco_utils import get_coco_api_from_dataset
from .coco_eval import CocoEvaluator
import train_utils.distributed_utils as utils


def train_one_epoch(model, optimizer, data_loader, device, epoch, num_epochs,
                    print_freq=50, warmup=False, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0 and warmup is True:  # 当训练第一轮（epoch=0）时，启用warmup训练方式，可理解为热身训练
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    mloss = torch.zeros(1).to(device)  # mean losses
    for i, [images, targets] in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        # 混合精度训练上下文管理器，如果在CPU环境中不起任何作用
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, epoch, num_epochs, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purpose
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()
        # 记录训练损失
        mloss = (mloss * i + loss_value) / (i + 1)  # update mean losses

        if not math.isfinite(loss_value):  # 当计算的损失为无穷大时停止训练
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:  # 第一轮使用warmup训练方式
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        now_lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=now_lr)

    return mloss, now_lr


@torch.no_grad()
def evaluate(model, data_loader, device, epoch, num_epochs):

    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test: "

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for image, targets in metric_logger.log_every(data_loader, 100, header):
        image = list(img.to(device) for img in image)

        # 当使用CPU时，跳过GPU相关指令
        if device != torch.device("cpu"):
            torch.cuda.synchronize(device)

        model_time = time.time()
        outputs = model(image, epoch, num_epochs)
        # 做一个锚框的可视化
        # if epoch > num_epochs+5-5:
        #     image1 = image[0].squeeze(0).cpu().numpy() * 256
        #     # print(outputs[0])
        #     boxes = outputs[0]['boxes']
        #     fig, ax = plt.subplots()
        #     plt.imshow(image1)
        #     for i in range(boxes.shape[0]):
        #         xmin, ymin, xmax, ymax = boxes[i].cpu().numpy()
        #         rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='r',
        #                                  facecolor='none')
        #         ax.add_patch(rect)
        #     plt.show()

        outputs = [{k: v.to(cpu_device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}

        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    coco_info = coco_evaluator.coco_eval[iou_types[0]].stats.tolist()  # numpy to list

    # Add code to get PR curve data
    class_pr_info = []
    coco_eval = coco_evaluator.coco_eval['bbox']
    
    # 检查数据集类型，如果是VOC数据集，使用class_dict获取类别信息
    if hasattr(data_loader.dataset, 'coco'):
        cat_ids = data_loader.dataset.coco.getCatIds()
        cat_names = [data_loader.dataset.coco.loadCats(cat_id)[0]['name'] for cat_id in cat_ids]
    else:
        # 对于VOC数据集，使用class_dict
        class_dict = data_loader.dataset.class_dict
        cat_ids = list(class_dict.values())
        cat_names = list(class_dict.keys())
    
    # Precision is a 5-dim array: [T, R, K, A, M]
    # T: 10 IoU thresholds
    # R: 101 recall thresholds
    # K: num_classes
    # A: 4 area ranges
    # M: 3 max detections
    precision = coco_eval.eval['precision']

    # Get PR curve for each class at IoU=0.5
    # IoU index 0 corresponds to IoU=0.5
    # Area index 0 corresponds to all areas
    # Max detection index 2 corresponds to 100 detections
    for i, cat_id in enumerate(cat_ids):
        pr = precision[0, :, i, 0, 2]
        recall = coco_eval.params.recThrs
        class_pr_info.append((pr.tolist(), recall.tolist(), cat_names[i]))

    return coco_info, class_pr_info


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    return iou_types
