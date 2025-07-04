import os

def get_image_names(directory):
    # 获取目录下所有文件
    files = os.listdir(directory)
    
    # 过滤出jpg文件并去掉扩展名
    image_names = [os.path.splitext(f)[0] for f in files if f.endswith('.jpg')]
    
    # 按名称排序
    image_names.sort()
    
    return image_names

if __name__ == "__main__":
    # 指定图像目录路径
    image_dir = r"E:\FRCNN\faster_rcnn\VOCdevkit\Mission_hole\images\Missing_hole"
    
    # 获取图像名称列表
    names = get_image_names(image_dir)
    
    # 将图像名称保存到txt文件
    output_file = "image_names.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        for name in names:
            f.write(name + '\n')
    
    print(f"图像名称已保存到 {output_file}")
    print(f"总共有 {len(names)} 个图像") 