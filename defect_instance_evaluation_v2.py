#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
缺陷实例级评估策略测试代码（伪代码实现版）
评估方法严格按照用户伪代码实现，IoU阈值0.99，实例级匹配
"""

import os
import xml.etree.ElementTree as ET
import numpy as np
import json
import torch
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor
from network_files import FasterRCNN, AnchorsGenerator
from backbone import densenet_literature17, densenet_literature20, densenet_mod1, densenet_mod2, densenet_mod3, densenet_mod4, densenet_mod5, densenet_mod6
from backbone import densenet121yuan, densenet169
from my_dataset import VOCDataSet
import transforms
import time
import datetime
import cv2

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

def parse_enhanced_xml(xml_path, target_class_id):
    with open('./pascal_crack.json', 'r') as f:
        class_dict = json.load(f)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    defect_instances = {}
    for obj in root.findall('object'):
        name_elem = obj.find('name')
        if name_elem is None:
            continue
        name = name_elem.text
        class_id = class_dict.get(name)
        if class_id != target_class_id:
            continue
        instance_id_elem = obj.find('instance_id')
        if instance_id_elem is None:
            continue
        instance_id = instance_id_elem.text
        bndbox = obj.find('bndbox')
        if bndbox is None:
            continue
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)
        box = [xmin, ymin, xmax, ymax]
        if instance_id not in defect_instances:
            defect_instances[instance_id] = []
        defect_instances[instance_id].append(box)
    return defect_instances

def visualize_fp_boxes(image_path, fp_boxes, save_path):
    """
    在图片上绘制FP预测框并保存
    :param image_path: 原始图片路径
    :param fp_boxes: FP框列表，每个为[xmin, ymin, xmax, ymax]
    :param save_path: 保存路径
    """
    image = cv2.imread(image_path)
    for box in fp_boxes:
        xmin, ymin, xmax, ymax = map(int, box)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)  # 红色框
    cv2.imwrite(save_path, image)

def defect_instance_eval_v2(predictions, targets, iou_threshold=0.3, val_dataset=None, results_dir=None, visualize=True):
    """
    按伪代码实现缺陷实例级评估
    """
    # 读取类别映射
    with open('./pascal_crack.json', 'r') as f:
        class_dict = json.load(f)
    class_ids = set()
    for target in targets:
        if 'labels' in target:
            class_ids.update(target['labels'])
    all_class_results = {}
    for class_id in class_ids:
        # 收集所有预测框（按置信度降序）
        pred_boxes = []  # [(box, score, img_idx)]
        for img_idx, pred in enumerate(predictions):
            if 'labels' in pred and 'boxes' in pred and 'scores' in pred:
                class_mask = pred['labels'] == class_id
                class_pred_boxes = pred['boxes'][class_mask]
                class_pred_scores = pred['scores'][class_mask]
                for box, score in zip(class_pred_boxes, class_pred_scores):
                    pred_boxes.append((box, score, img_idx))
        pred_boxes = sorted(pred_boxes, key=lambda x: x[1], reverse=True)
        P = len(pred_boxes)
        # 收集所有GT实例
        gt_instances = {}  # {unique_instance_id: [boxes]}
        instance_id_map = {}  # {gt_box_idx: unique_instance_id}
        for img_idx, target in enumerate(targets):
            if 'xml_path' in target:
                xml_path = target['xml_path']
                img_gt_instances = parse_enhanced_xml(xml_path, class_id)
                for instance_id, boxes in img_gt_instances.items():
                    unique_instance_id = f"img_{img_idx}_{instance_id}"
                    gt_instances[unique_instance_id] = boxes
        # 统计所有GT框及其所属实例
        G = []  # [(box, unique_instance_id)]
        for instance_id, boxes in gt_instances.items():
            for box in boxes:
                G.append((box, instance_id))
        D = list(gt_instances.keys())
        N = len(D)
        TP_list = [0] * P
        FP_list = [0] * P
        fp_boxes_per_image = {}  # {img_idx: [box1, box2, ...]}
        matched = set()
        for j, (bj, score, img_idx) in enumerate(pred_boxes):
            # 计算与所有GT框的IoU
            ious = [calculate_iou(bj, g) for g, _ in G]
            has_overlap = any(iou > 0 for iou in ious)
            # 候选集C: defect_id(g) for g in G if IoU(bj, g) >= iou_threshold
            C = set()
            for idx, (g, instance_id) in enumerate(G):
                if ious[idx] >= iou_threshold:
                    C.add(instance_id)
            V = C - matched
            if V:
                best_k = None
                best_iou = -1
                for k in V:
                    for g in gt_instances[k]:
                        iou = calculate_iou(bj, g)
                        if iou > best_iou:
                            best_iou = iou
                            best_k = k
                TP_list[j] = 1
                matched.add(best_k)
            elif not has_overlap:
                FP_list[j] = 1
                if img_idx not in fp_boxes_per_image:
                    fp_boxes_per_image[img_idx] = []
                fp_boxes_per_image[img_idx].append(bj)
            # 否则既不是TP也不是FP
        TP_cum = np.cumsum(TP_list)
        FP_cum = np.cumsum(FP_list)
        precisions = TP_cum / (TP_cum + FP_cum + 1e-8)
        recalls = TP_cum / N if N > 0 else np.zeros_like(TP_cum)
        # AP插值法
        ap = 0.0
        for t in np.linspace(0, 1, 101):
            p = precisions[recalls >= t].max() if np.any(recalls >= t) else 0
            ap += p / 101
        final_precision = precisions[-1] if len(precisions) > 0 else 0
        final_recall = recalls[-1] if len(recalls) > 0 else 0
        f1_score = 2 * final_precision * final_recall / (final_precision + final_recall) if (final_precision + final_recall) > 0 else 0
        all_class_results[class_id] = {
            'total_gt_instances': N,
            'total_pred_instances': P,
            'tp': int(TP_cum[-1]) if len(TP_cum) > 0 else 0,
            'fp': int(FP_cum[-1]) if len(FP_cum) > 0 else 0,
            'fn': N - (int(TP_cum[-1]) if len(TP_cum) > 0 else 0),
            'precision': final_precision,
            'recall': final_recall,
            'f1_score': f1_score,
            'ap': ap
        }
        # 可视化TP和FP
        if visualize and val_dataset is not None and results_dir is not None:
            for img_idx in set(list(fp_boxes_per_image.keys()) + list(range(len(targets)))):
                # 获取图片路径
                xml_path = targets[img_idx]['xml_path']
                tree = ET.parse(xml_path)
                root = tree.getroot()
                filename = root.find('filename').text
                img_path = os.path.join(val_dataset.img_root, filename)
                # 收集FP框
                fp_boxes = fp_boxes_per_image.get(img_idx, [])
                # 收集TP框
                tp_boxes = []
                for j, (bj, score, idx) in enumerate(pred_boxes):
                    if TP_list[j] == 1 and idx == img_idx:
                        tp_boxes.append(bj)
                # 合成可视化
                def visualize_tp_fp_boxes(image_path, tp_boxes, fp_boxes, save_path):
                    image = cv2.imread(image_path)
                    for box in tp_boxes:
                        xmin, ymin, xmax, ymax = map(int, box)
                        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)  # 绿色TP
                    for box in fp_boxes:
                        xmin, ymin, xmax, ymax = map(int, box)
                        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)  # 红色FP
                    cv2.imwrite(save_path, image)
                save_path = os.path.join(results_dir, f'tp_fp_vis_class{class_id}_img{img_idx}.jpg')
                visualize_tp_fp_boxes(img_path, tp_boxes, fp_boxes, save_path)
    return all_class_results

def create_model(num_classes, model_name):
    import torchvision.models as models
    from torchvision.models.feature_extraction import create_feature_extractor
    from backbone import densenet_literature17, densenet_literature20, densenet_mod1, densenet_mod2, densenet_mod3, densenet_mod4, densenet_mod5, densenet_mod6
    from backbone import densenet121yuan, densenet169
    from network_files import FasterRCNN, AnchorsGenerator
    if model_name == "densenet_literature17":
        model = densenet_literature17()
        model.features.conv0 = torch.nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1, bias=False)
        with torch.no_grad():
            test_input = torch.randn(1, 3, 608, 608)
            test_output = model.features(test_input)
            actual_channels = test_output.shape[1]
        model.features.add_module("final_conv", torch.nn.Conv2d(actual_channels, 128, kernel_size=1, stride=1, bias=False))
        return_nodes = {'features.final_conv': '0'}
        backbone = create_feature_extractor(model, return_nodes=return_nodes)
        backbone.out_channels = 128
    elif model_name == "densenet_literature20":
        model = densenet_literature20()
        model.features.conv0 = torch.nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1, bias=False)
        with torch.no_grad():
            test_input = torch.randn(1, 3, 608, 608)
            test_output = model.features(test_input)
            actual_channels = test_output.shape[1]
        model.features.add_module("final_conv", torch.nn.Conv2d(actual_channels, 128, kernel_size=1, stride=1, bias=False))
        return_nodes = {'features.final_conv': '0'}
        backbone = create_feature_extractor(model, return_nodes=return_nodes)
        backbone.out_channels = 128
    elif model_name == "densenet_mod1":
        model = densenet_mod1()
        model.features.conv0 = torch.nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1, bias=False)
        with torch.no_grad():
            test_input = torch.randn(1, 3, 608, 608)
            test_output = model.features(test_input)
            actual_channels = test_output.shape[1]
        model.features.add_module("final_conv", torch.nn.Conv2d(actual_channels, 128, kernel_size=1, stride=1, bias=False))
        return_nodes = {'features.final_conv': '0'}
        backbone = create_feature_extractor(model, return_nodes=return_nodes)
        backbone.out_channels = 128
    elif model_name == "densenet_mod2":
        model = densenet_mod2()
        model.features.conv0 = torch.nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1, bias=False)
        with torch.no_grad():
            test_input = torch.randn(1, 3, 608, 608)
            test_output = model.features(test_input)
            actual_channels = test_output.shape[1]
        model.features.add_module("final_conv", torch.nn.Conv2d(actual_channels, 128, kernel_size=1, stride=1, bias=False))
        return_nodes = {'features.final_conv': '0'}
        backbone = create_feature_extractor(model, return_nodes=return_nodes)
        backbone.out_channels = 128
    elif model_name == "densenet_mod3":
        model = densenet_mod3()
        model.features.conv0 = torch.nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1, bias=False)
        with torch.no_grad():
            test_input = torch.randn(1, 3, 608, 608)
            test_output = model.features(test_input)
            actual_channels = test_output.shape[1]
        model.features.add_module("final_conv", torch.nn.Conv2d(actual_channels, 128, kernel_size=1, stride=1, bias=False))
        return_nodes = {'features.final_conv': '0'}
        backbone = create_feature_extractor(model, return_nodes=return_nodes)
        backbone.out_channels = 128
    elif model_name == "densenet_mod4":
        model = densenet_mod4()
        model.features.conv0 = torch.nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1, bias=False)
        with torch.no_grad():
            test_input = torch.randn(1, 3, 608, 608)
            test_output = model.features(test_input)
            actual_channels = test_output.shape[1]
        model.features.add_module("final_conv", torch.nn.Conv2d(actual_channels, 128, kernel_size=1, stride=1, bias=False))
        return_nodes = {'features.final_conv': '0'}
        backbone = create_feature_extractor(model, return_nodes=return_nodes)
        backbone.out_channels = 128
    elif model_name == "densenet_mod5":
        model = densenet_mod5()
        model.features.conv0 = torch.nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1, bias=False)
        with torch.no_grad():
            test_input = torch.randn(1, 3, 608, 608)
            test_output = model.features(test_input)
            actual_channels = test_output.shape[1]
        model.features.add_module("final_conv", torch.nn.Conv2d(actual_channels, 128, kernel_size=1, stride=1, bias=False))
        return_nodes = {'features.final_conv': '0'}
        backbone = create_feature_extractor(model, return_nodes=return_nodes)
        backbone.out_channels = 128
    elif model_name == "densenet_mod6":
        model = densenet_mod6()
        model.features.conv0 = torch.nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1, bias=False)
        with torch.no_grad():
            test_input = torch.randn(1, 3, 608, 608)
            test_output = model.features(test_input)
            actual_channels = test_output.shape[1]
        model.features.add_module("final_conv", torch.nn.Conv2d(actual_channels, 128, kernel_size=1, stride=1, bias=False))
        return_nodes = {'features.final_conv': '0'}
        backbone = create_feature_extractor(model, return_nodes=return_nodes)
        backbone.out_channels = 128
    elif model_name == "densenet121yuan":
        model = densenet121yuan()
        model.features.conv0 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        with torch.no_grad():
            test_input = torch.randn(1, 3, 608, 608)
            test_output = model.features(test_input)
            actual_channels = test_output.shape[1]
        model.features.add_module("final_conv", torch.nn.Conv2d(actual_channels, 128, kernel_size=1, stride=1, bias=False))
        return_nodes = {'features.final_conv': '0'}
        backbone = create_feature_extractor(model, return_nodes=return_nodes)
        backbone.out_channels = 128
    elif model_name == "densenet169":
        model = densenet169()
        model.features.conv0 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        with torch.no_grad():
            test_input = torch.randn(1, 3, 608, 608)
            test_output = model.features(test_input)
            actual_channels = test_output.shape[1]
        model.features.add_module("final_conv", torch.nn.Conv2d(actual_channels, 128, kernel_size=1, stride=1, bias=False))
        return_nodes = {'features.final_conv': '0'}
        backbone = create_feature_extractor(model, return_nodes=return_nodes)
        backbone.out_channels = 128
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=False)
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
        model = models.efficientnet_b0(pretrained=False)
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
        model = models.shufflenet_v2_x1_0(pretrained=False)
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
    anchor_generator = AnchorsGenerator(sizes=((14),), aspect_ratios=((1.0),))
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=[7, 7], sampling_ratio=2)
    model = FasterRCNN(backbone=backbone,
                      num_classes=num_classes,
                      rpn_anchor_generator=anchor_generator,
                      box_roi_pool=roi_pooler)
    return model

def evaluate_model(model, data_loader, device, weights_path):
    print(f"加载模型权重: {weights_path}")
    checkpoint = torch.load(weights_path, map_location=device)
    if 'model' in checkpoint:
        print("检测到完整训练检查点，加载模型权重...")
        model.load_state_dict(checkpoint['model'])
    else:
        print("检测到纯模型权重，直接加载...")
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    all_predictions = []
    all_targets = []
    print("开始模型推理...")
    with torch.no_grad():
        for i, (images, targets) in enumerate(data_loader):
            if i % 10 == 0:
                print(f"处理第 {i+1} 批图像...")
            images = list(img.to(device) for img in images)
            outputs = model(images, 0, 1)
            for output, target in zip(outputs, targets):
                pred_boxes = output['boxes'].cpu().numpy()
                pred_scores = output['scores'].cpu().numpy()
                pred_labels = output['labels'].cpu().numpy()
                all_predictions.append({
                    'boxes': pred_boxes,
                    'scores': pred_scores,
                    'labels': pred_labels
                })
                target_dict = {
                    'boxes': target['boxes'].cpu().numpy(),
                    'labels': target['labels'].cpu().numpy()
                }
                if 'xml_path' in target:
                    target_dict['xml_path'] = target['xml_path']
                if 'instance_ids' in target:
                    target_dict['instance_ids'] = target['instance_ids']
                all_targets.append(target_dict)
    print(f"推理完成，共处理 {len(all_predictions)} 张图像")
    return all_predictions, all_targets

def main():
    print("=" * 60)
    print("缺陷实例级评估策略测试（伪代码实现版）")
    print("=" * 60)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    data_transform = {"val": transforms.Compose([transforms.ToTensor()])}
    VOC_root = "./"
    val_dataset = VOCDataSet(VOC_root, "2012", data_transform["val"], "val.txt")
    val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                                batch_size=1,
                                                shuffle=False,
                                                pin_memory=False,
                                                num_workers=2,
                                                collate_fn=val_dataset.collate_fn)
    print(f"验证集大小: {len(val_dataset)} 张图像")
    num_classes = 2

    # 只测试一个模型，手动指定
    model_name = "densenet169"  # 你要测试的主干模型名
    weights_path = "save_weights_densenet169/densenet169-model-504.pth"  # 对应权重路径

    if not os.path.exists(weights_path):
        print(f"权重文件不存在: {weights_path}")
        return
    results_dir = f"defect_instance_eval_v2_results_{model_name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    os.makedirs(results_dir, exist_ok=True)
    print(f"创建模型: {model_name}")
    model = create_model(num_classes, model_name)
    all_predictions, all_targets = evaluate_model(model, val_data_loader, device, weights_path)
    print("\n" + "=" * 60)
    print(f"缺陷实例级评估指标（{model_name}，伪代码实现）")
    print("=" * 60)
    results = defect_instance_eval_v2(all_predictions, all_targets, iou_threshold=0.8, val_dataset=val_dataset, results_dir=results_dir)
    summary_file = os.path.join(results_dir, "evaluation_summary_v2.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"缺陷实例级评估结果汇总（{model_name}，伪代码实现）\n")
        f.write("=" * 50 + "\n\n")
        for class_id, res in results.items():
            f.write(f"类别 {class_id} 结果:\n")
            f.write(f"  AP: {res['ap']:.4f}\n")
            f.write(f"  Precision: {res['precision']:.4f}\n")
            f.write(f"  Recall: {res['recall']:.4f}\n")
            f.write(f"  F1-Score: {res['f1_score']:.4f}\n")
            f.write(f"  TP: {res['tp']}, FP: {res['fp']}, FN: {res['fn']}\n\n")
    print(f"\n评估完成！所有结果保存在: {results_dir}")
    print(f"综合结果文件: {summary_file}")

if __name__ == '__main__':
    main() 