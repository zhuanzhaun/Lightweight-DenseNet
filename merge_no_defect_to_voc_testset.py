import os
import shutil

# 路径配置
# 新无缺陷图片和xml标签目录
no_defect_img_dir = r'E:/FRCNN/pP10BNoCrack'
no_defect_xml_dir = os.path.join(no_defect_img_dir, 'Annotations_empty')
# VOC测试集目录
voc_root = r'E:/FRCNN/faster_rcnn 对比实验/VOC2012'
voc_img_dir = os.path.join(voc_root, 'JPEGImages')
voc_xml_dir = os.path.join(voc_root, 'Annotations')
val_txt_path = os.path.join(voc_root, 'ImageSets', 'Main', 'val.txt')

# 1. 拷贝图片
img_files = [f for f in os.listdir(no_defect_img_dir) if f.lower().endswith('.bmp')]
for img_name in img_files:
    src = os.path.join(no_defect_img_dir, img_name)
    dst = os.path.join(voc_img_dir, img_name)
    shutil.copy2(src, dst)
    print(f'已复制图片: {img_name}')

# 2. 拷贝xml标签
xml_files = [f for f in os.listdir(no_defect_xml_dir) if f.lower().endswith('.xml')]
for xml_name in xml_files:
    src = os.path.join(no_defect_xml_dir, xml_name)
    dst = os.path.join(voc_xml_dir, xml_name)
    shutil.copy2(src, dst)
    print(f'已复制标签: {xml_name}')

# 3. 更新val.txt
# 读取原有val.txt内容
with open(val_txt_path, 'r', encoding='utf-8') as f:
    val_lines = [line.strip() for line in f.readlines() if line.strip()]
# 新增无缺陷图片名（不带扩展名）
new_names = [os.path.splitext(f)[0] for f in img_files]
# 去重合并
all_names = val_lines + [n for n in new_names if n not in val_lines]
# 写回val.txt
with open(val_txt_path, 'w', encoding='utf-8') as f:
    for name in all_names:
        f.write(name + '\n')
print(f'val.txt已更新，共{len(all_names)}张图片。') 