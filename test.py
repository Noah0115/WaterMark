import math
import zlib
import cv2
from blind_watermark import WaterMark


def encryption():
    """
    嵌入水印到图片
    :return: 水印的长度（字节数）
    """
    # 读取长文本
    with open('input_gbk.txt', 'r', encoding='gbk') as file:
        long_text = file.read()

    # 读取原始图片
    original_image = cv2.imread("input_img/test.jpg")
    img_shape = original_image.shape[:2]  # 图片尺寸

    # 计算图片中的水印容量
    img_ori_byte = (img_shape[0] * img_shape[1]) // 64
    byte = bin(int(zlib.compress(long_text.encode('GBK')).hex(), base=16))[2:]

    # 如果水印容量大于图片容量，则调整图片大小以容纳水印
    if img_ori_byte < len(byte):
        mult = round(math.sqrt((len(byte) / img_ori_byte) * 2), 2)
        target_size = (round(img_shape[1] * mult), round(img_shape[0] * mult))
        img_resized = cv2.resize(original_image, target_size)
        cv2.imwrite("input_img/new_test.jpg", img_resized)

        # 在调整后的图片中嵌入水印
        watermark_embedder = WaterMark(password_wm=1, password_img=1)
        watermark_embedder.read_img("input_img/new_test.jpg")
        watermark_embedder.read_wm(long_text, mode='str')
        watermark_embedder.embed('output_img/output1.jpg')

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


if __name__ == '__main__':
    wm_shape = encryption()
    watermark = extract_watermark_from_image('output_img/output1.jpg', wm_shape)
