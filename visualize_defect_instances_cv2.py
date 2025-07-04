import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from collections import defaultdict

def parse_enhanced_xml(xml_path):
    """
    解析增强的XML文件，提取缺陷实例信息
    返回: {instance_id: {'boxes': [[xmin, ymin, xmax, ymax], ...], 'attributes': {...}, 'group': ...}}
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    instances_info = {}
    for obj in root.findall('object'):
        bndbox = obj.find('bndbox')
        xmin = int(float(bndbox.find('xmin').text))
        ymin = int(float(bndbox.find('ymin').text))
        xmax = int(float(bndbox.find('xmax').text))
        ymax = int(float(bndbox.find('ymax').text))
        instance_id = obj.find('instance_id').text
        instance_group = obj.find('instance_group').text
        instance_attrs = obj.find('instance_attributes')
        attributes = {}
        if instance_attrs is not None:
            for attr in instance_attrs:
                attributes[attr.tag] = attr.text
        if instance_id not in instances_info:
            instances_info[instance_id] = {
                'boxes': [],
                'attributes': attributes,
                'group': instance_group
            }
        instances_info[instance_id]['boxes'].append([xmin, ymin, xmax, ymax])
    return instances_info

def random_color(seed):
    np.random.seed(seed)
    return tuple(int(x) for x in np.random.randint(0, 255, 3))

def visualize_defect_instances_cv2(image_path, xml_path, output_path=None, show_attributes=True):
    """
    用cv2可视化缺陷实例，保存图片
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return
    instances_info = parse_enhanced_xml(xml_path)
    color_map = {}
    for idx, instance_id in enumerate(instances_info.keys()):
        color_map[instance_id] = random_color(idx+1)
    # 绘制每个实例
    for idx, (instance_id, info) in enumerate(instances_info.items()):
        color = color_map[instance_id]
        for j, box in enumerate(info['boxes']):
            xmin, ymin, xmax, ymax = box
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
            if j == 0:
                label = f"{instance_id}"
                cv2.putText(image, label, (xmin, max(ymin-10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    # 标题和统计信息
    instance_count = len(instances_info)
    total_boxes = sum(len(info['boxes']) for info in instances_info.values())
    title = f"Instances: {instance_count}, Boxes: {total_boxes}"
    cv2.putText(image, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
    # 可选：显示属性信息（左上角）
    if show_attributes:
        y0 = 50
        for instance_id, info in instances_info.items():
            attrs = info['attributes']
            attr_str = f"{instance_id}: n={len(info['boxes'])} sev={attrs.get('severity','N/A')} ori={attrs.get('orientation','N/A')}"
            cv2.putText(image, attr_str, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_map[instance_id], 1)
            y0 += 20
    if output_path:
        cv2.imwrite(output_path, image)
        print(f"保存可视化结果: {output_path}")
    else:
        cv2.imshow('defect instances', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def create_instance_summary_cv2(xml_dir, output_file=None):
    summary = {}
    total_instances = 0
    total_boxes = 0
    xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]
    for xml_file in xml_files:
        xml_path = os.path.join(xml_dir, xml_file)
        instances_info = parse_enhanced_xml(xml_path)
        instance_count = len(instances_info)
        box_count = sum(len(info['boxes']) for info in instances_info.values())
        summary[xml_file] = {
            'instance_count': instance_count,
            'box_count': box_count,
            'instances': instances_info
        }
        total_instances += instance_count
        total_boxes += box_count
        print(f"{xml_file}: {instance_count} instances, {box_count} boxes")
    print(f"Total: {total_instances} instances, {total_boxes} boxes")
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            for xml_file, info in summary.items():
                f.write(f"{xml_file}: {info['instance_count']} instances, {info['box_count']} boxes\n")
            f.write(f"Total: {total_instances} instances, {total_boxes} boxes\n")
        print(f"统计摘要已保存到: {output_file}")
    return summary

def create_batch_visualization_cv2(image_dir, xml_dir, output_dir, max_images=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]
    if max_images is None:
        max_images = len(xml_files)
    processed_count = 0
    for xml_file in xml_files[:max_images]:
        image_name = xml_file.replace('.xml', '.bmp')
        image_path = os.path.join(image_dir, image_name)
        if os.path.exists(image_path):
            xml_path = os.path.join(xml_dir, xml_file)
            output_path = os.path.join(output_dir, f"vis_{xml_file.replace('.xml', '.png')}")
            try:
                visualize_defect_instances_cv2(image_path, xml_path, output_path, show_attributes=False)
                processed_count += 1
            except Exception as e:
                print(f"处理 {xml_file} 出错: {e}")
        else:
            print(f"未找到图像: {image_path}")
    print(f"批量可视化完成，共处理 {processed_count} 张图像，结果保存在: {output_dir}")

if __name__ == '__main__':
    # 示例：单张
    # visualize_defect_instances_cv2('VOC2012/JPEGImages/1101.bmp', 'VOC2012/Annotations_Enhanced/1101.xml')
    # 示例：批量
    create_batch_visualization_cv2('VOC2012/JPEGImages', 'VOC2012/Annotations_Enhanced', 'visualization_results_cv2', max_images=None)
 