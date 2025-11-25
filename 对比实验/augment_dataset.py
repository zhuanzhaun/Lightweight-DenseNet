"""
数据增强脚本：对训练集整体应用四种干扰增强方法，生成四个新的干扰数据集
1. 高斯噪声 (Gaussian Noise)
2. 光照扰动 (Illumination Perturbation)
3. 模拟油污 (Simulated Oil Stain)
4. 混合以上三种
"""

import os
import sys
import numpy as np
from PIL import Image
from lxml import etree
import random
import shutil

# 尝试导入tqdm，如果没有则使用简单的进度显示
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # 简单的进度显示函数
    def tqdm(iterable, desc=""):
        return iterable

# 尝试导入cv2，如果失败则使用numpy实现高斯模糊
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("警告: 未安装opencv-python，将使用numpy实现高斯模糊（效果可能略差）")

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_xml_to_dict(xml):
    """解析XML为字典"""
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


def gaussian_noise(image, mean=0, std=10):
    """
    高斯噪声增强
    
    Args:
        image: PIL Image对象 (RGB格式)
        mean: 噪声均值，默认0
        std: 噪声标准差，默认10（控制噪声强度）
    
    Returns:
        noisy_image: 添加噪声后的PIL Image对象
    """
    # 转换为numpy数组
    img_array = np.array(image, dtype=np.float32)
    
    # 生成与图像尺寸相同的高斯噪声
    h, w, c = img_array.shape
    noise = np.random.normal(mean, std, (h, w, c))
    
    # 添加噪声
    noisy_image = img_array + noise
    
    # 限制值在[0, 255]范围内
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    # 转换回PIL Image
    return Image.fromarray(noisy_image)


def illumination_perturbation(image, alpha_range=(0.7, 1.3), beta_range=(-30, 30)):
    """
    光照扰动增强
    
    Args:
        image: PIL Image对象 (RGB格式)
        alpha_range: 对比度增益范围 (gain)，默认(0.7, 1.3)
        beta_range: 亮度偏置范围 (bias)，默认(-30, 30)
    
    Returns:
        perturbed_image: 光照扰动后的PIL Image对象
    """
    # 转换为numpy数组
    img_array = np.array(image, dtype=np.float32)
    
    # 随机生成alpha和beta
    alpha = np.random.uniform(alpha_range[0], alpha_range[1])
    beta = np.random.uniform(beta_range[0], beta_range[1])
    
    # 线性变换: output = alpha * image + beta
    perturbed_image = alpha * img_array + beta
    
    # 限制值在[0, 255]范围内
    perturbed_image = np.clip(perturbed_image, 0, 255).astype(np.uint8)
    
    # 转换回PIL Image
    return Image.fromarray(perturbed_image)


def simulated_oil_stain(image, num_stains=1, stain_intensity=0.6):
    """
    模拟油污增强
    
    Args:
        image: PIL Image对象 (RGB格式)
        num_stains: 油污数量，默认1
        stain_intensity: 油污强度（透明度），范围[0, 1]，默认0.6
    
    Returns:
        stained_image: 添加油污后的PIL Image对象
    """
    # 转换为numpy数组
    img_array = np.array(image, dtype=np.float32)
    h, w, c = img_array.shape
    
    # 创建mask（全零矩阵）
    mask = np.zeros((h, w), dtype=np.float32)
    
    # 定义油污颜色（低亮度的灰色/蓝黑色）
    stain_color = np.array([30, 30, 30], dtype=np.float32)
    
    # 在mask上绘制随机形状的油污区域
    for _ in range(num_stains):
        # 随机选择油污中心位置
        center_x = np.random.randint(w // 4, 3 * w // 4)
        center_y = np.random.randint(h // 4, 3 * h // 4)
        
        # 随机选择油污大小
        radius_x = np.random.randint(min(w, h) // 8, min(w, h) // 4)
        radius_y = np.random.randint(min(w, h) // 8, min(w, h) // 4)
        
        # 创建椭圆mask
        y, x = np.ogrid[:h, :w]
        ellipse_mask = ((x - center_x) / radius_x) ** 2 + ((y - center_y) / radius_y) ** 2 <= 1
        
        # 添加一些随机变形（使用高斯模糊模拟不规则形状）
        if ellipse_mask.sum() > 0:
            # 将布尔mask转换为浮点数
            stain_mask = ellipse_mask.astype(np.float32)
            # 应用高斯模糊使边缘更自然
            if HAS_CV2:
                stain_mask = cv2.GaussianBlur(stain_mask, (21, 21), 0)
            else:
                # 使用scipy实现高斯模糊
                try:
                    from scipy.ndimage import gaussian_filter
                    stain_mask = gaussian_filter(stain_mask, sigma=5)
                except ImportError:
                    # 如果scipy也不可用，使用简单的平滑（numpy实现）
                    kernel_size = 15
                    pad_size = kernel_size // 2
                    # 填充边界
                    padded = np.pad(stain_mask, pad_size, mode='edge')
                    # 简单的均值滤波
                    smoothed = np.zeros_like(stain_mask)
                    for i in range(h):
                        for j in range(w):
                            smoothed[i, j] = np.mean(padded[i:i+kernel_size, j:j+kernel_size])
                    stain_mask = smoothed
            # 更新总mask（取最大值，避免重叠区域过度增强）
            mask = np.maximum(mask, stain_mask * stain_intensity)
    
    # 将mask扩展到3通道
    mask_3d = np.expand_dims(mask, axis=2).repeat(3, axis=2)
    
    # 混合原图和油污颜色: output = image * (1 - mask) + stain_color * mask
    stained_image = img_array * (1 - mask_3d) + stain_color * mask_3d
    
    # 限制值在[0, 255]范围内
    stained_image = np.clip(stained_image, 0, 255).astype(np.uint8)
    
    # 转换回PIL Image
    return Image.fromarray(stained_image)


def mixed_augmentation(image, 
                     gaussian_std=10, 
                     alpha_range=(0.8, 1.2), 
                     beta_range=(-20, 20),
                     num_stains=1,
                     stain_intensity=0.5):
    """
    混合增强：同时应用高斯噪声、光照扰动和模拟油污
    
    Args:
        image: PIL Image对象 (RGB格式)
        gaussian_std: 高斯噪声标准差
        alpha_range: 光照扰动对比度范围
        beta_range: 光照扰动亮度范围
        num_stains: 油污数量
        stain_intensity: 油污强度
    
    Returns:
        mixed_image: 混合增强后的PIL Image对象
    """
    # 先应用光照扰动
    image = illumination_perturbation(image, alpha_range, beta_range)
    
    # 再应用高斯噪声
    image = gaussian_noise(image, mean=0, std=gaussian_std)
    
    # 最后应用模拟油污
    image = simulated_oil_stain(image, num_stains, stain_intensity)
    
    return image


def copy_xml_file(src_xml_path, dst_xml_path):
    """
    复制XML标注文件
    
    Args:
        src_xml_path: 源XML文件路径
        dst_xml_path: 目标XML文件路径
    """
    if os.path.exists(src_xml_path):
        # 确保目标目录存在
        dst_dir = os.path.dirname(dst_xml_path)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir, exist_ok=True)
        # 复制XML文件
        shutil.copy2(src_xml_path, dst_xml_path)


def process_dataset(voc_root, year="2012", txt_names=["train.txt", "val.txt"], 
                   output_base_dir="augmented_datasets"):
    """
    处理整个数据集，应用四种增强方法并生成四个新的数据集文件夹
    
    Args:
        voc_root: VOC数据集根目录
        year: 数据集年份
        txt_names: 列表文件名列表（如 ["train.txt", "val.txt"]）
        output_base_dir: 输出数据集的基础目录
    """
    # 构建数据集路径
    root = os.path.join(voc_root, "VOCdevkit", f"VOC{year}")
    img_root = os.path.join(root, "JPEGImages")
    annotations_root = os.path.join(root, "Annotations")
    
    # 收集所有需要处理的图片名称（从train.txt和val.txt）
    all_image_names = set()  # 使用set避免重复
    
    for txt_name in txt_names:
        txt_path = os.path.join(root, "ImageSets", "Main", txt_name)
        
        if not os.path.exists(txt_path):
            print(f"警告: 未找到文件 {txt_path}，跳过")
            continue
        
        with open(txt_path, 'r') as f:
            image_names = [line.strip() for line in f.readlines() if line.strip()]
            all_image_names.update(image_names)
            print(f"从 {txt_name} 读取到 {len(image_names)} 张图片")
    
    if len(all_image_names) == 0:
        raise ValueError(f"没有找到任何图片信息")
    
    # 转换为列表并排序
    image_names = sorted(list(all_image_names))
    print(f"\n总共需要处理 {len(image_names)} 张图片（去重后）")
    
    # 创建输出目录结构
    output_base = os.path.abspath(output_base_dir)
    augmentation_types = [
        "01_gaussian_noise",
        "02_illumination_perturbation",
        "03_oil_stain",
        "04_mixed_augmentation"
    ]
    
    # 为每种增强类型创建数据集目录结构
    for aug_type in augmentation_types:
        aug_output_dir = os.path.join(output_base, aug_type, "VOCdevkit", f"VOC{year}")
        aug_img_dir = os.path.join(aug_output_dir, "JPEGImages")
        aug_annotations_dir = os.path.join(aug_output_dir, "Annotations")
        aug_imagesets_dir = os.path.join(aug_output_dir, "ImageSets", "Main")
        
        os.makedirs(aug_img_dir, exist_ok=True)
        os.makedirs(aug_annotations_dir, exist_ok=True)
        os.makedirs(aug_imagesets_dir, exist_ok=True)
        
        print(f"已创建目录: {aug_output_dir}")
    
    # 处理每张图片
    print("\n开始处理图片...")
    for idx, image_name in enumerate(tqdm(image_names, desc="处理进度")):
        # 尝试读取XML获取图片文件名
        xml_path = os.path.join(annotations_root, f"{image_name}.xml")
        if os.path.exists(xml_path):
            with open(xml_path, 'rb') as fid:
                xml_str = fid.read()
            xml = etree.fromstring(xml_str)
            data = parse_xml_to_dict(xml)["annotation"]
            filename = data.get("filename", f"{image_name}.jpg")
        else:
            # 如果没有XML，尝试常见的图片格式
            filename = None
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']:
                test_path = os.path.join(img_root, f"{image_name}{ext}")
                if os.path.exists(test_path):
                    filename = f"{image_name}{ext}"
                    break
            
            if filename is None:
                print(f"警告: 跳过图片 {image_name}，未找到图片文件")
                continue
        
        img_path = os.path.join(img_root, filename)
        
        if not os.path.exists(img_path):
            print(f"警告: 跳过图片 {img_path}，文件不存在")
            continue
        
        # 加载原始图片
        try:
            original_image = Image.open(img_path)
            # 确保图像是RGB模式
            if original_image.mode != 'RGB':
                original_image = original_image.convert('RGB')
        except Exception as e:
            print(f"警告: 无法加载图片 {img_path}: {e}")
            continue
        
        # 获取图片基础名称（不含扩展名）
        base_name = os.path.splitext(filename)[0]
        img_ext = os.path.splitext(filename)[1]
        
        # 应用四种增强方法并保存
        # 1. 高斯噪声
        try:
            gaussian_img = gaussian_noise(original_image.copy(), mean=0, std=10)
            gaussian_path = os.path.join(output_base, "01_gaussian_noise", "VOCdevkit", 
                                       f"VOC{year}", "JPEGImages", filename)
            gaussian_img.save(gaussian_path, "PNG" if img_ext.lower() == '.png' else "JPEG", quality=95)
            # 复制XML文件
            if os.path.exists(xml_path):
                copy_xml_file(xml_path, os.path.join(output_base, "01_gaussian_noise", "VOCdevkit",
                                                   f"VOC{year}", "Annotations", f"{image_name}.xml"))
        except Exception as e:
            print(f"警告: 处理高斯噪声增强失败 {filename}: {e}")
        
        # 2. 光照扰动
        try:
            illumination_img = illumination_perturbation(original_image.copy(), 
                                                       alpha_range=(0.7, 1.3), 
                                                       beta_range=(-30, 30))
            illumination_path = os.path.join(output_base, "02_illumination_perturbation", "VOCdevkit",
                                           f"VOC{year}", "JPEGImages", filename)
            illumination_img.save(illumination_path, "PNG" if img_ext.lower() == '.png' else "JPEG", quality=95)
            # 复制XML文件
            if os.path.exists(xml_path):
                copy_xml_file(xml_path, os.path.join(output_base, "02_illumination_perturbation", "VOCdevkit",
                                                   f"VOC{year}", "Annotations", f"{image_name}.xml"))
        except Exception as e:
            print(f"警告: 处理光照扰动增强失败 {filename}: {e}")
        
        # 3. 模拟油污
        try:
            oil_stain_img = simulated_oil_stain(original_image.copy(), num_stains=2, stain_intensity=0.6)
            oil_stain_path = os.path.join(output_base, "03_oil_stain", "VOCdevkit",
                                        f"VOC{year}", "JPEGImages", filename)
            oil_stain_img.save(oil_stain_path, "PNG" if img_ext.lower() == '.png' else "JPEG", quality=95)
            # 复制XML文件
            if os.path.exists(xml_path):
                copy_xml_file(xml_path, os.path.join(output_base, "03_oil_stain", "VOCdevkit",
                                                   f"VOC{year}", "Annotations", f"{image_name}.xml"))
        except Exception as e:
            print(f"警告: 处理模拟油污增强失败 {filename}: {e}")
        
        # 4. 混合增强
        try:
            mixed_img = mixed_augmentation(original_image.copy(),
                                         gaussian_std=10,
                                         alpha_range=(0.8, 1.2),
                                         beta_range=(-20, 20),
                                         num_stains=1,
                                         stain_intensity=0.5)
            mixed_path = os.path.join(output_base, "04_mixed_augmentation", "VOCdevkit",
                                     f"VOC{year}", "JPEGImages", filename)
            mixed_img.save(mixed_path, "PNG" if img_ext.lower() == '.png' else "JPEG", quality=95)
            # 复制XML文件
            if os.path.exists(xml_path):
                copy_xml_file(xml_path, os.path.join(output_base, "04_mixed_augmentation", "VOCdevkit",
                                                   f"VOC{year}", "Annotations", f"{image_name}.xml"))
        except Exception as e:
            print(f"警告: 处理混合增强失败 {filename}: {e}")
    
    # 复制train.txt和val.txt到每个增强数据集的ImageSets/Main目录
    print("\n复制数据集列表文件...")
    for aug_type in augmentation_types:
        aug_imagesets_dir = os.path.join(output_base, aug_type, "VOCdevkit", 
                                        f"VOC{year}", "ImageSets", "Main")
        
        for txt_name in txt_names:
            src_txt_path = os.path.join(root, "ImageSets", "Main", txt_name)
            if os.path.exists(src_txt_path):
                dst_txt_path = os.path.join(aug_imagesets_dir, txt_name)
                shutil.copy2(src_txt_path, dst_txt_path)
                print(f"已复制 {txt_name} 到 {aug_type}")
    
    print("\n" + "=" * 60)
    print("数据集增强完成！")
    print("=" * 60)
    print(f"\n生成的增强数据集位于: {output_base}")
    print("\n数据集结构:")
    for aug_type in augmentation_types:
        print(f"  - {aug_type}/VOCdevkit/VOC{year}/")
        print(f"      - JPEGImages/  (增强后的图片)")
        print(f"      - Annotations/  (原始标注文件)")
        print(f"      - ImageSets/Main/  (数据集列表文件)")
        for txt_name in txt_names:
            print(f"          - {txt_name}")


def main():
    """主函数"""
    print("=" * 60)
    print("数据集整体增强脚本")
    print("=" * 60)
    
    # 设置随机种子以便结果可复现
    np.random.seed(42)
    random.seed(42)
    
    # 数据路径配置
    voc_root = "E:/FRCNN/faster_rcnn 对比实验/"  # VOCdevkit根目录
    year = "2012"
    txt_names = ["train.txt", "val.txt"]  # 同时处理训练集和验证集
    output_base_dir = "augmented_datasets"  # 输出数据集的基础目录
    
    try:
        # 处理数据集（训练集和验证集）
        process_dataset(voc_root, year, txt_names, output_base_dir)
        
    except FileNotFoundError as e:
        print(f"\n错误: {e}")
        print("\n提示:")
        print("1. 请确保数据集路径正确")
        print("2. 请确保存在训练集列表文件")
        print("3. 如果数据集不存在，请先准备数据集")
        return
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()

