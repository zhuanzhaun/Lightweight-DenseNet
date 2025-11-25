import os
import datetime
import sys
import torch
from ultralytics import YOLO
import json

sys.setrecursionlimit(100000000)

def train_yolov8_nano():
    """
    使用与train_desnet.py相同的训练数据训练YOLOv8-nano模型
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training YOLOv8-nano using {device.type} device.")
    
    # 清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU memory before training: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        print(f"GPU memory cached: {torch.cuda.memory_reserved()/1024**3:.2f} GB")

    # 用来保存训练结果的文件
    results_file = f"results_YOLOv8-nano_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.txt"

    # 检查保存权重文件夹是否存在，不存在则创建
    save_dir = f"save_weights_YOLOv8-nano"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 数据路径配置（与train_desnet.py保持一致）
    VOC_root = "./"  # VOCdevkit
    year = "2012"
    
    if os.path.exists(os.path.join(VOC_root, "VOCdevkit")) is False:
        raise FileNotFoundError("VOCdevkit dose not in path:'{}'.".format(VOC_root))

    # 读取类别信息
    json_file = './pascal_crack.json'
    assert os.path.exists(json_file), "{} file not exist.".format(json_file)
    with open(json_file, 'r') as f:
        class_dict = json.load(f)
    
    # 获取类别名称列表（YOLO格式需要类别名称列表）
    class_names = list(class_dict.keys())
    num_classes = len(class_names)
    print(f"类别数量: {num_classes}")
    print(f"类别名称: {class_names}")

    # 构建数据集路径
    voc_dataset_path = os.path.join(VOC_root, "VOCdevkit", f"VOC{year}")
    train_txt_path = os.path.join(voc_dataset_path, "ImageSets", "Main", "train.txt")
    val_txt_path = os.path.join(voc_dataset_path, "ImageSets", "Main", "val.txt")
    
    # 由于YOLOv8默认使用YOLO格式（txt标注），而当前数据是VOC格式（XML标注）
    # 我们需要将VOC格式转换为YOLO格式
    yolo_labels_path = os.path.join(voc_dataset_path, "labels")
    if not os.path.exists(yolo_labels_path) or len(os.listdir(yolo_labels_path)) == 0:
        print("检测到VOC格式数据，开始转换为YOLO格式...")
        convert_voc_to_yolo(voc_dataset_path, train_txt_path, val_txt_path, class_dict)
        print("数据格式转换完成！")
    else:
        print(f"检测到已存在的YOLO格式标注文件，跳过转换。")
    
    # 创建YOLO格式的数据集配置文件（YAML格式）
    dataset_yaml_path = "yolov8_dataset.yaml"
    yaml_content = f"""# YOLOv8 数据集配置文件
# 基于 train_desnet.py 的数据路径配置

path: {os.path.abspath(voc_dataset_path)}  # 数据集根目录
train: JPEGImages  # 训练图片目录（相对于path，YOLOv8会自动查找对应的labels目录）
val: JPEGImages    # 验证图片目录（相对于path，YOLOv8会自动查找对应的labels目录）

# 类别信息
names:
"""
    for idx, class_name in enumerate(class_names):
        yaml_content += f"  {idx}: {class_name}\n"
    
    # 保存YAML配置文件
    with open(dataset_yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    print(f"数据集配置文件已保存: {dataset_yaml_path}")

    # 初始化YOLOv8-nano模型
    model = YOLO('yolov8n.pt')  # 加载预训练的nano模型
    print("YOLOv8-nano 模型已加载")

    # 训练参数配置（参考train_desnet.py的训练策略）
    # 第一阶段：冻结backbone（YOLOv8通过freeze参数实现）
    # 第二阶段：解冻全部层进行训练
    
    # 第一阶段训练：冻结backbone（前5个epoch）
    print("\n开始第一阶段训练（冻结backbone，5个epoch）...")
    model.train(
        data=dataset_yaml_path,           # 数据集配置文件
        epochs=5,                          # 第一阶段训练5个epoch
        imgsz=608,                         # 输入图像尺寸（与train_desnet.py一致）
        batch=1,                           # 批次大小（与train_desnet.py一致）
        device=device,                     # 训练设备
        project=save_dir,                  # 项目保存目录
        name='YOLOv8-nano_stage1',         # 第一阶段训练运行名称
        exist_ok=True,                     # 允许覆盖已存在的项目
        pretrained=True,                   # 使用预训练权重
        optimizer='SGD',                   # 优化器（与train_desnet.py一致）
        lr0=0.001,                         # 初始学习率（与train_desnet.py一致）
        momentum=0.9,                      # 动量（与train_desnet.py一致）
        weight_decay=0.0002,               # 权重衰减（第一阶段）
        warmup_epochs=3,                   # 预热轮数
        warmup_momentum=0.8,               # 预热动量
        warmup_bias_lr=0.1,                # 预热偏置学习率
        box=7.5,                           # 边界框损失权重
        cls=0.5,                           # 分类损失权重
        dfl=1.5,                           # DFL损失权重
        save=True,                         # 保存检查点
        save_period=5,                     # 每N轮保存一次
        val=True,                          # 训练期间进行验证
        plots=True,                        # 保存训练曲线图
        verbose=True,                      # 详细输出
        freeze=10,                         # 冻结前10层（相当于冻结backbone）
    )

    # 第二阶段训练：解冻全部层
    print("\n开始第二阶段训练（解冻全部层，45个epoch）...")
    # 获取第一阶段训练的最佳权重文件
    stage1_weights = os.path.join(save_dir, "YOLOv8-nano_stage1", "weights", "best.pt")
    if not os.path.exists(stage1_weights):
        # 如果best.pt不存在，尝试使用last.pt
        stage1_weights = os.path.join(save_dir, "YOLOv8-nano_stage1", "weights", "last.pt")
        if not os.path.exists(stage1_weights):
            print("警告: 未找到第一阶段训练的权重文件，将从头开始第二阶段训练")
            stage1_weights = None
    
    # 如果找到第一阶段权重，加载它
    if stage1_weights and os.path.exists(stage1_weights):
        model = YOLO(stage1_weights)
        print(f"已加载第一阶段训练权重: {stage1_weights}")
    
    model.train(
        data=dataset_yaml_path,           # 数据集配置文件
        epochs=45,                         # 第二阶段训练45个epoch
        imgsz=608,                         # 输入图像尺寸
        batch=1,                           # 批次大小
        device=device,                     # 训练设备
        project=save_dir,                  # 项目保存目录
        name='YOLOv8-nano_stage2',         # 第二阶段训练运行名称
        exist_ok=True,                     # 允许覆盖已存在的项目
        optimizer='SGD',                   # 优化器
        lr0=0.001,                         # 初始学习率
        momentum=0.9,                      # 动量
        weight_decay=0.00005,              # 权重衰减（第二阶段，与train_desnet.py一致）
        warmup_epochs=3,                   # 预热轮数
        warmup_momentum=0.8,               # 预热动量
        warmup_bias_lr=0.1,                # 预热偏置学习率
        box=7.5,                           # 边界框损失权重
        cls=0.5,                           # 分类损失权重
        dfl=1.5,                           # DFL损失权重
        save=True,                         # 保存检查点
        save_period=10,                    # 每N轮保存一次
        val=True,                          # 训练期间进行验证
        plots=True,                        # 保存训练曲线图
        verbose=True,                      # 详细输出
        freeze=0,                          # 解冻所有层
    )

    print(f"\n训练完成！模型保存在: {save_dir}")
    print(f"训练结果文件: {results_file}")


def convert_voc_to_yolo(voc_dataset_path, train_txt_path, val_txt_path, class_dict):
    """
    将VOC格式（XML）转换为YOLO格式（TXT）
    
    Args:
        voc_dataset_path: VOC数据集路径
        train_txt_path: 训练集列表文件路径
        val_txt_path: 验证集列表文件路径
        class_dict: 类别字典
    """
    from lxml import etree
    from PIL import Image
    
    # 创建labels目录
    labels_dir = os.path.join(voc_dataset_path, "labels")
    os.makedirs(labels_dir, exist_ok=True)
    
    # 读取类别名称到索引的映射
    class_name_to_id = {name: idx for idx, name in enumerate(class_dict.keys())}
    
    # 处理训练集和验证集
    for txt_path in [train_txt_path, val_txt_path]:
        if not os.path.exists(txt_path):
            continue
            
        with open(txt_path, 'r') as f:
            image_names = [line.strip() for line in f.readlines() if line.strip()]
        
        for image_name in image_names:
            # 读取XML标注文件
            xml_path = os.path.join(voc_dataset_path, "Annotations", f"{image_name}.xml")
            if not os.path.exists(xml_path):
                continue
            
            # 解析XML
            try:
                with open(xml_path, 'rb') as fid:
                    xml_str = fid.read()
                xml = etree.fromstring(xml_str)
                data = parse_xml_to_dict(xml)["annotation"]
            except Exception as e:
                print(f"警告: 无法解析XML文件 {xml_path}: {e}，跳过")
                continue
            
            # 读取XML获取图片尺寸（优先从XML读取，避免读取图片文件）
            try:
                size_data = data.get("size", {})
                if size_data and "width" in size_data and "height" in size_data:
                    img_width = int(size_data.get("width", 0))
                    img_height = int(size_data.get("height", 0))
                    if img_width == 0 or img_height == 0:
                        raise ValueError("XML中的尺寸信息无效")
                else:
                    # 如果XML中没有size信息，则读取图片
                    img_path = os.path.join(voc_dataset_path, "JPEGImages", f"{image_name}.jpg")
                    if not os.path.exists(img_path):
                        # 尝试其他格式
                        for ext in ['.png', '.jpeg', '.JPG', '.PNG', '.jpg']:
                            alt_path = os.path.join(voc_dataset_path, "JPEGImages", f"{image_name}{ext}")
                            if os.path.exists(alt_path):
                                img_path = alt_path
                                break
                        else:
                            print(f"警告: 找不到图片文件 {image_name}，跳过")
                            continue
                    img = Image.open(img_path)
                    img_width, img_height = img.size
            except Exception as e:
                print(f"警告: 无法获取图片尺寸 {image_name}: {e}，跳过")
                continue
            
            # 创建YOLO格式的标注文件
            yolo_label_path = os.path.join(labels_dir, f"{image_name}.txt")
            with open(yolo_label_path, 'w') as f:
                if "object" in data:
                    for obj in data["object"]:
                        xmin = float(obj["bndbox"]["xmin"])
                        xmax = float(obj["bndbox"]["xmax"])
                        ymin = float(obj["bndbox"]["ymin"])
                        ymax = float(obj["bndbox"]["ymax"])
                        
                        # 检查边界框有效性
                        if xmax <= xmin or ymax <= ymin:
                            continue
                        
                        # 转换为YOLO格式（归一化的中心坐标和宽高）
                        x_center = ((xmin + xmax) / 2.0) / img_width
                        y_center = ((ymin + ymax) / 2.0) / img_height
                        width = (xmax - xmin) / img_width
                        height = (ymax - ymin) / img_height
                        
                        # 获取类别ID
                        class_name = obj["name"]
                        if class_name in class_name_to_id:
                            class_id = class_name_to_id[class_name]
                            # YOLO格式: class_id x_center y_center width height
                            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


def parse_xml_to_dict(xml):
    """
    解析XML为字典（从my_dataset.py复制）
    """
    if len(xml) == 0:
        return {xml.tag: xml.text}
    
    result = {}
    for child in xml:
        child_result = parse_xml_to_dict(child)
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}


def main():
    """
    主函数
    """
    print("=" * 50)
    print("开始训练 YOLOv8-nano 模型")
    print("使用与 train_desnet.py 相同的训练数据")
    print("=" * 50)
    
    try:
        train_yolov8_nano()
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

