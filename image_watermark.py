import math
import zlib
import cv2
from blind_watermark import WaterMark
import os

def encryption(ori_img_name,watermark_filename):
    """
    嵌入水印到图片
    :return: 水印的长度（字节数）
    """
    # 读取长文本
    with open(watermark_filename, 'r', encoding='gbk') as file:
        long_text = file.read()

    # 读取原始图片
    original_image = cv2.imread("input_img/"+ori_img_name)
    img_shape = original_image.shape[:2]  # 图片尺寸

    # 计算图片中的水印容量
    img_ori_byte = (img_shape[0] * img_shape[1]) // 64
    byte = bin(int(zlib.compress(long_text.encode('GBK')).hex(), base=16))[2:]

    # 如果水印容量大于图片容量，则调整图片大小以容纳水印
    if img_ori_byte < len(byte):
        mult = round(math.sqrt((len(byte) / img_ori_byte) * 2), 2)
        target_size = (round(img_shape[1] * mult), round(img_shape[0] * mult))
        img_resized = cv2.resize(original_image, target_size)
        cv2.imwrite("input_img/new_"+ori_img_name, img_resized)

        # 在调整后的图片中嵌入水印
        watermark_embedder = WaterMark(password_wm=1, password_img=1)
        watermark_embedder.read_img("input_img/new_"+ori_img_name)
        watermark_embedder.read_wm(long_text, mode='str')
        watermark_embedder.embed("output_img/output_"+ori_img_name)
        return len(byte)

    watermark_embedder = WaterMark(password_wm=1, password_img=1)
    watermark_embedder.read_img("input_img/" + ori_img_name)
    watermark_embedder.read_wm(long_text, mode='str')
    watermark_embedder.embed("output_img/output_" + ori_img_name)
    return len(byte)


def extract_watermark_from_image(embedded_image_path, wm_shape):
    """
    从已嵌入的图片读取水印信息
    :param embedded_image_path: 嵌入水印的图片路径
    :param wm_shape: 水印的长度（字节数）
    :return: 提取的水印信息
    """
    watermark_extractor = WaterMark(password_wm=1, password_img=1)
    extracted_watermark = watermark_extractor.extract(embedded_image_path, wm_shape=wm_shape, mode="str")
    print(extracted_watermark)
    return extracted_watermark


def print_directory_tree(path, indent="", excluded_folders=None):
    """
        输出项目目录结构函数
    :param path:
    :param indent:
    :param excluded_folders:
    :return:
    """
    if excluded_folders is None:
        excluded_folders = []

    items = os.listdir(path)
    for i, item in enumerate(items):
        item_path = os.path.join(path, item)
        is_last = i == len(items) - 1

        if os.path.isdir(item_path):
            if item not in excluded_folders:
                print(indent + ("└── " if is_last else "├── ") + item)
                new_indent = indent + ("    " if is_last else "│   ")
                print_directory_tree(item_path, new_indent, excluded_folders)
        else:
            print(indent + ("└── " if is_last else "├── ") + item)


if __name__ == '__main__':
    warter_mark_filename = "input_gbk.txt"
    input_filename = "test.jpg"
    wm_shape = encryption("test.jpg",warter_mark_filename)
    watermark = extract_watermark_from_image("output_img/output_"+input_filename, wm_shape)
    # watermark = extract_watermark_from_image('input_img/test2.jpg', wm_shape)


    #--------- 输出项目目录 树形结构 -----------------
    # starting_path = "./"
    # exclude_folders = ["venv",".idea","__pycache__"]  # 要排除的文件夹名称列表
    # print_directory_tree(starting_path, excluded_folders=exclude_folders)


