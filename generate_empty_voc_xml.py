import os
import cv2
from xml.dom.minidom import Document
import re

# 输入图片文件夹
img_dir = r'E:/FRCNN/pP10BNoCrack'
# 输出xml文件夹
xml_dir = os.path.join(img_dir, 'Annotations_empty')
os.makedirs(xml_dir, exist_ok=True)

def remove_chinese(text):
    return re.sub(r'[\u4e00-\u9fa5]', '', text)

img_files = [f for f in os.listdir(img_dir) if f.lower().endswith('.bmp')]

for img_name in img_files:
    # 检查并重命名含中文的图片
    new_img_name = remove_chinese(img_name)
    if new_img_name != img_name:
        old_path = os.path.join(img_dir, img_name)
        new_path = os.path.join(img_dir, new_img_name)
        os.rename(old_path, new_path)
        print(f'已重命名: {img_name} -> {new_img_name}')
        img_name = new_img_name
    img_path = os.path.join(img_dir, img_name)
    img = cv2.imread(img_path)
    if img is None:
        print(f'无法读取图片: {img_path}')
        continue
    height, width = img.shape[:2]
    depth = img.shape[2] if len(img.shape) == 3 else 1
    # 创建xml
    doc = Document()
    annotation = doc.createElement('annotation')
    doc.appendChild(annotation)

    folder = doc.createElement('folder')
    folder.appendChild(doc.createTextNode('JPEGImages'))
    annotation.appendChild(folder)

    filename = doc.createElement('filename')
    filename.appendChild(doc.createTextNode(img_name))
    annotation.appendChild(filename)

    path = doc.createElement('path')
    path.appendChild(doc.createTextNode(img_path))
    annotation.appendChild(path)

    source = doc.createElement('source')
    database = doc.createElement('database')
    database.appendChild(doc.createTextNode('hulan'))
    source.appendChild(database)
    annotation.appendChild(source)

    size = doc.createElement('size')
    width_elem = doc.createElement('width')
    width_elem.appendChild(doc.createTextNode(str(width)))
    size.appendChild(width_elem)
    height_elem = doc.createElement('height')
    height_elem.appendChild(doc.createTextNode(str(height)))
    size.appendChild(height_elem)
    depth_elem = doc.createElement('depth')
    depth_elem.appendChild(doc.createTextNode(str(depth)))
    size.appendChild(depth_elem)
    annotation.appendChild(size)

    segmented = doc.createElement('segmented')
    segmented.appendChild(doc.createTextNode('0'))
    annotation.appendChild(segmented)

    # 不添加<object>，表示无缺陷

    # 保存xml
    xml_name = os.path.splitext(img_name)[0] + '.xml'
    xml_path = os.path.join(xml_dir, xml_name)
    with open(xml_path, 'w', encoding='utf-8') as f:
        doc.writexml(f, indent='\t', addindent='\t', newl='\n', encoding='utf-8')
    print(f'已生成: {xml_path}') 