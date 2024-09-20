# 隐水印(watermark)使用文档

## 目录结构

#### 项目目录

```
├── attr_test.py				//	对图片攻击测试py文件
├── blind_watermark				//	隐水印库文件夹
│   ├── att.py					//	图片攻击测试
│   ├── blind_watermark.py		        //	WaterMark类
│   ├── bwm_core.py				//	WaterMarkCore类
│   ├── cli_tools.py			        //	命令行运行方式及参数说明
│   ├── pool.py					//	图片池化
│   ├── recover.py				//	工具类
│   ├── version.py				//	版本说明
│   ├── __init__.py				//	初始化文件
├── image_watermark.py			        //	图片嵌入水印测试py文件
├── input.txt					//	输入语料
├── input_gbk.txt				//	输入语料(GBK编码)
├── input_img					//	待加密图片文件夹
│   ├── new_test.jpg			        //	嵌入过程中对原图修改后的图片文件
│   └── test.jpg				//	待加密原始图片
├── output_img					//	嵌入水印后生成图片的文件夹
│   └── output1.jpg				//	已嵌入水印图片
├── requirements.txt			        //	项目依赖
├── watermark_attr.py			        //	图片攻击类
```

## 类及函数说明

### 图片嵌入水印(image_watermark.py)：

在image_watermark.py中，定义函数encryption()、extract_watermark_from_image()。

encryption()函数为嵌入水印功能的实现。首先从本地目录读取input.txt水印文本，保存到long_text变量当中，之后读取原始图片test.jpg，使用opencv读取图片，获取到图片的参数，然后分别计算原始图片最多可容纳水印信息的容量与long_text转换为字节数的大小，之后将图片信息容量与文本字节容量进行比较，如果文本字节容量超过到图片信息容量，那么就将原始图片进行像素扩大以保证能够完整的嵌入全部的信息。最后调用WaterMark类进行水印嵌入操作。

```python
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
```



### 攻击测试类(watermark_attr.py):

在watermark_attr.py中，定义了GanWatermark类，默认情况下嵌入的水印为“深圳杯数学建模挑战赛”其中功能包含：不攻击情况下提取水印、压缩攻击并提取水印、变换格式攻击并提取水印、缩放攻击并提取水印、旋转攻击并提取水印、椒盐攻击并提取水印、遮挡攻击并提取水印、亮度攻击并提取水印。

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np
from blind_watermark import WaterMark, att


class GanWatermark:
    def __init__(self):
        """
        初始化GanWatermark类
        """
        self.ori_img = cv2.imread('input_img/test.jpg', flags=cv2.IMREAD_UNCHANGED)
        self.wm = '深圳杯数学建模挑战赛'
        self.ori_img_shape = self.ori_img.shape[:2]
        self.h, self.w = self.ori_img_shape
        self.bwm = WaterMark(password_img=1, password_wm=1)
        self.bwm.read_img(img=self.ori_img)
        self.bwm.read_wm(self.wm, mode='str')
        self.embed_img = self.bwm.embed("output_img/attr_test.jpg",)
        self.len_wm = len(self.bwm.wm_bit)
        print('水印长度为： {len_wm}'.format(len_wm=self.len_wm))

    def unGan(self):
        """
        不攻击的情况下提取水印
        """
        un_gan_wm_extract = self.bwm.extract('output_img/attr_test.jpg', wm_shape=self.len_wm, mode='str')
        print("不攻击的提取结果：", un_gan_wm_extract)

    def compressGan(self):
        """
        压缩攻击并提取水印
        """
        success, compressed_img = cv2.imencode('.jpg', self.embed_img, [cv2.IMWRITE_JPEG_QUALITY, 50])
        with open('output_img/compressed.jpg', 'wb') as f:
            f.write(compressed_img)
        compress_gan_wm_extract = self.bwm.extract('output_img/compressed.jpg', wm_shape=self.len_wm, mode='str')
        print("压缩攻击的提取结果：", compress_gan_wm_extract)

    def resetFormatGan(self):
        """
        变换格式攻击并提取水印
        """
        cv2.imwrite('output_img/reset_format.png', self.embed_img)
        wm_extract = self.bwm.extract('output_img/reset_format.png', wm_shape=self.len_wm, mode='str')
        print("变换格式攻击的提取结果：", wm_extract)

    def resizeGan(self):
        """
        缩放攻击并提取水印
        """
        image_attacked = att.resize_att(input_img=self.embed_img, out_shape=(400, 300))
        image_recover = att.resize_att(input_img=image_attacked, out_shape=self.ori_img_shape[::-1])
        cv2.imwrite('output_img/resize.jpg', image_attacked)
        cv2.imwrite('output_img/resize_recover.jpg', image_recover)
        resize_extract = self.bwm.extract(embed_img=image_recover, wm_shape=self.len_wm, mode='str')
        print("缩放攻击后的提取结果：", resize_extract)

    def rotateGan(self):
        """
        旋转攻击并提取水印
        """
        angle = 60
        image_attacked = att.rot_att(input_img=self.embed_img, angle=angle)
        image_recover = att.rot_att(input_img=image_attacked, output_file_name='output_img/rotate_recover.jpg',
                                    angle=-angle)
        rotate_extract = self.bwm.extract(embed_img=image_recover, wm_shape=self.len_wm, mode='str')
        print(f"旋转攻击angle={angle}后的提取结果：", rotate_extract)

    def saltGan(self):
        """
        椒盐攻击并提取水印
        """
        ratio = 0.05
        image_attacked = att.salt_pepper_att(input_img=self.embed_img, output_file_name='output_img/salt.jpg',
                                             ratio=ratio)
        salt_extract = self.bwm.extract(embed_img=image_attacked, wm_shape=self.len_wm, mode='str')
        print(f"椒盐攻击ratio={ratio}后的提取结果：", salt_extract)

    def shelterGan(self):
        """
        遮挡攻击并提取水印
        """
        n = 60
        image_attacked = att.shelter_att(input_img=self.embed_img, output_file_name='output_img/shelter.jpg', ratio=0.1,
                                         n=n)
        shelter_extract = self.bwm.extract(embed_img=image_attacked, wm_shape=self.len_wm, mode='str')
        print(f"遮挡攻击{n}次后的提取结果：", shelter_extract)
        assert self.wm == shelter_extract, '提取水印和原水印不一致'

    def brightGan(self):
        """
        亮度攻击并提取水印
        """
        img_recover = att.bright_att(input_img=self.embed_img, output_file_name='output_img/bright_recover.jpg',
                                     ratio=1.2)
        wm_extract = self.bwm.extract(embed_img=img_recover, wm_shape=self.len_wm, mode='str')
        print("亮度攻击后的提取结果：", wm_extract)
        assert np.all(self.wm == wm_extract), '提取水印和原水印不一致'
```

### 测试攻击(attr_test.py)：

在attr_test.py文件中，实例化GanWatermark类，分别调用类中攻击函数：

```python
import watermark_attr

def gan_test():
    gan_obj = watermark_attr.GanWatermark()
    gan_obj.unGan()             # 不攻击的情况下，提取水印
    gan_obj.saltGan()           # 椒盐攻击
    gan_obj.compressGan()       # 压缩攻击
    gan_obj.brightGan()         # 亮度攻击
    gan_obj.resetFormatGan()    # 格式攻击
    gan_obj.resizeGan()         # 缩放攻击
    gan_obj.rotateGan()         # 旋转攻击
    gan_obj.shelterGan()        # 遮挡攻击
if __name__ == '__main__':
    gan_test()
```

### 核心类(bwm_core.py)：

WaterMarkCore类其中定义了嵌入图片的核心算法以及嵌入过程，此部分**基于blind_watermark库原作者**的代码上修改为符合需求的功能。

```python
#!/usr/bin/env python3
# coding=utf-8
# @Time    : 2021/12/17
# @Author  : github.com/guofei9987
import numpy as np
import copy
import cv2
from pywt import dwt2, idwt2
from .pool import AutoPool
import math


class WaterMarkCore:
    def __init__(self, password_img=1, mode='common', processes=None):
        self.block_shape = np.array([4, 4])  # 原
        # self.block_shape = np.array([2, 2])
        self.password_img = password_img
        self.d1, self.d2 = 36, 20  # d1/d2 越大鲁棒性越强,但输出图片的失真越大 原
        # init data
        self.img, self.img_YUV = None, None  # self.img 是原图，self.img_YUV 对像素做了加白偶数化
        self.ca, self.hvd, = [np.array([])] * 3, [np.array([])] * 3  # 每个通道 dct 的结果
        self.ca_block = [np.array([])] * 3  # 每个 channel 存一个四维 array，代表四维分块后的结果
        self.ca_part = [np.array([])] * 3  # 四维分块后，有时因不整除而少一部分，self.ca_part 是少这一部分的 self.ca
        self.wm_size, self.block_num = 0, 0  # 水印的长度，原图片可插入信息的个数
        self.pool = AutoPool(mode=mode, processes=processes)

    def init_block_index(self):
        self.block_num = self.ca_block_shape[0] * self.ca_block_shape[1]
        assert self.wm_size < self.block_num, IndexError(
            '最多可嵌入{}kb信息，多于水印的{}kb信息，溢出'.format(self.block_num / 1000, self.wm_size / 1000))
        # self.part_shape 是取整后的ca二维大小,用于嵌入时忽略右边和下面对不齐的细条部分。
        self.part_shape = self.ca_block_shape[:2] * self.block_shape
        self.block_index = [(i, j) for i in range(self.ca_block_shape[0]) for j in range(self.ca_block_shape[1])]

    def read_img(self, filename):
        # 读入图片->YUV化->加白边使像素变偶数->四维分块
        img = cv2.imread(filename)
        if img is None:
            raise IOError("image file '{filename}' not read".format(filename=filename))

        self.img = img.astype(np.float32)
        self.img_shape = self.img.shape[:2]

        # 如果不是偶数，那么补上白边，Y（明亮度）UV（颜色）
        self.img_YUV = cv2.copyMakeBorder(cv2.cvtColor(self.img, cv2.COLOR_BGR2YUV),
                                          0, self.img.shape[0] % 2, 0, self.img.shape[1] % 2,
                                          cv2.BORDER_CONSTANT, value=(0, 0, 0))

        self.ca_shape = [(i + 1) // 2 for i in self.img_shape]

        self.ca_block_shape = (self.ca_shape[0] // self.block_shape[0], self.ca_shape[1] // self.block_shape[1],
                               self.block_shape[0], self.block_shape[1])
        strides = 4 * np.array([self.ca_shape[1] * self.block_shape[0], self.block_shape[1], self.ca_shape[1], 1])

        for channel in range(3):
            self.ca[channel], self.hvd[channel] = dwt2(self.img_YUV[:, :, channel], 'haar')
            # 转为4维度
            self.ca_block[channel] = np.lib.stride_tricks.as_strided(self.ca[channel].astype(np.float32),
                                                                     self.ca_block_shape, strides)
    def read_img_arr(self, img):
        # 处理透明图
        self.alpha = None
        if img.shape[2] == 4:
            if img[:, :, 3].min() < 255:
                self.alpha = img[:, :, 3]
                img = img[:, :, :3]

        # 读入图片->YUV化->加白边使像素变偶数->四维分块
        self.img = img.astype(np.float32)
        self.img_shape = self.img.shape[:2]

        # 如果不是偶数，那么补上白边，Y（明亮度）UV（颜色）
        self.img_YUV = cv2.copyMakeBorder(cv2.cvtColor(self.img, cv2.COLOR_BGR2YUV),
                                          0, self.img.shape[0] % 2, 0, self.img.shape[1] % 2,
                                          cv2.BORDER_CONSTANT, value=(0, 0, 0))

        self.ca_shape = [(i + 1) // 2 for i in self.img_shape]

        self.ca_block_shape = (self.ca_shape[0] // self.block_shape[0], self.ca_shape[1] // self.block_shape[1],
                               self.block_shape[0], self.block_shape[1])
        strides = 4 * np.array([self.ca_shape[1] * self.block_shape[0], self.block_shape[1], self.ca_shape[1], 1])

        for channel in range(3):
            self.ca[channel], self.hvd[channel] = dwt2(self.img_YUV[:, :, channel], 'haar')
            # 转为4维度
            self.ca_block[channel] = np.lib.stride_tricks.as_strided(self.ca[channel].astype(np.float32),
                                                                     self.ca_block_shape, strides)
    def read_wm(self, wm_bit):
        self.wm_bit = wm_bit
        self.wm_size = wm_bit.size

    def block_add_wm(self, arg):
        block, shuffler, i = arg
        # dct->(flatten->加密->逆flatten)->svd->打水印->逆svd->(flatten->解密->逆flatten)->逆dct
        wm_1 = self.wm_bit[i % self.wm_size]
        block_dct = cv2.dct(block)

        # 加密（打乱顺序）
        block_dct_shuffled = block_dct.flatten()[shuffler].reshape(self.block_shape)
        U, s, V = np.linalg.svd(block_dct_shuffled)
        s[0] = (s[0] // self.d1 + 1 / 4 + 1 / 2 * wm_1) * self.d1
        if self.d2:
            s[1] = (s[1] // self.d2 + 1 / 4 + 1 / 2 * wm_1) * self.d2

        block_dct_flatten = np.dot(U, np.dot(np.diag(s), V)).flatten()
        block_dct_flatten[shuffler] = block_dct_flatten.copy()
        return cv2.idct(block_dct_flatten.reshape(self.block_shape))

    def embed(self):
        self.init_block_index()

        embed_ca = copy.deepcopy(self.ca)
        embed_YUV = [np.array([])] * 3

        self.idx_shuffle = random_strategy1(self.password_img, self.block_num,
                                            self.block_shape[0] * self.block_shape[1])
        for channel in range(3):
            tmp = self.pool.map(self.block_add_wm,
                                [(self.ca_block[channel][self.block_index[i]], self.idx_shuffle[i], i)
                                 for i in range(self.block_num)])

            for i in range(self.block_num):
                self.ca_block[channel][self.block_index[i]] = tmp[i]

            # 4维分块变回2维
            self.ca_part[channel] = np.concatenate(np.concatenate(self.ca_block[channel], 1), 1)
            # 4维分块时右边和下边不能整除的长条保留，其余是主体部分，换成 embed 之后的频域的数据
            embed_ca[channel][:self.part_shape[0], :self.part_shape[1]] = self.ca_part[channel]
            # 逆变换回去
            embed_YUV[channel] = idwt2((embed_ca[channel], self.hvd[channel]), "haar")

        # 合并3通道
        embed_img_YUV = np.stack(embed_YUV, axis=2)
        # 之前如果不是2的整数，增加了白边，这里去除掉
        embed_img_YUV = embed_img_YUV[:self.img_shape[0], :self.img_shape[1]]
        embed_img = cv2.cvtColor(embed_img_YUV, cv2.COLOR_YUV2BGR)
        embed_img = np.clip(embed_img, a_min=0, a_max=255)

        if self.alpha is not None:
            embed_img = cv2.merge([embed_img.astype(np.uint8), self.alpha])
        return embed_img

    def block_get_wm(self, args):
        block, shuffler = args
        # dct->flatten->加密->逆flatten->svd->解水印
        block_dct_shuffled = cv2.dct(block).flatten()[shuffler].reshape(self.block_shape)

        U, s, V = np.linalg.svd(block_dct_shuffled)
        wm = (s[0] % self.d1 > self.d1 / 2) * 1
        if self.d2:
            tmp = (s[1] % self.d2 > self.d2 / 2) * 1
            wm = (wm * 3 + tmp * 1) / 4
        return wm

    def extract_raw(self, img):
        # 每个分块提取 1 bit 信息
        self.read_img_arr(img=img)
        self.init_block_index()

        wm_block_bit = np.zeros(shape=(3, self.block_num))  # 3个channel，length 个分块提取的水印，全都记录下来

        self.idx_shuffle = random_strategy1(seed=self.password_img,
                                            size=self.block_num,
                                            block_shape=self.block_shape[0] * self.block_shape[1],  # 16
                                            )
        for channel in range(3):
            wm_block_bit[channel, :] = self.pool.map(self.block_get_wm,
                                                     [(self.ca_block[channel][self.block_index[i]], self.idx_shuffle[i])
                                                      for i in range(self.block_num)])
        return wm_block_bit

    def extract_avg(self, wm_block_bit):
        # 对循环嵌入+3个 channel 求平均
        wm_avg = np.zeros(shape=self.wm_size)
        for i in range(self.wm_size):
            wm_avg[i] = wm_block_bit[:, i::self.wm_size].mean()
        return wm_avg

    def extract(self, img, wm_shape):
        self.wm_size = np.array(wm_shape).prod()

        # 提取每个分块埋入的 bit：
        wm_block_bit = self.extract_raw(img=img)
        # 做平均：
        wm_avg = self.extract_avg(wm_block_bit)
        return wm_avg

    def extract_with_kmeans(self, img, wm_shape):
        wm_avg = self.extract(img=img, wm_shape=wm_shape)

        return one_dim_kmeans(wm_avg)


def one_dim_kmeans(inputs):
    threshold = 0
    e_tol = 10 ** (-6)
    center = [inputs.min(), inputs.max()]  # 1. 初始化中心点
    for i in range(300):
        threshold = (center[0] + center[1]) / 2
        is_class01 = inputs > threshold  # 2. 检查所有点与这k个点之间的距离，每个点归类到最近的中心
        center = [inputs[~is_class01].mean(), inputs[is_class01].mean()]  # 3. 重新找中心点
        if np.abs((center[0] + center[1]) / 2 - threshold) < e_tol:  # 4. 停止条件
            threshold = (center[0] + center[1]) / 2
            break

    is_class01 = inputs > threshold
    return is_class01


def random_strategy1(seed, size, block_shape):
    return np.random.RandomState(seed) \
        .random(size=(size, block_shape)) \
        .argsort(axis=1)


def random_strategy2(seed, size, block_shape):
    one_line = np.random.RandomState(seed) \
        .random(size=(1, block_shape)) \
        .argsort(axis=1)

    return np.repeat(one_line, repeats=size, axis=0)
```

## 运行

### 使用方法

水印嵌入：运行**image_watermark.py**文件，将对**input_img/test.jpg**进行嵌入水印，嵌入水印后的图片保存在**output_img/output_test.jpg**(运行截图仅显示部分输出内容)

<img src="https://immich.lyh27.top/api/assets/d74388f5-f5fe-4bbc-a108-3f7e7fd5b32a/thumbnail?size=preview&key=25V6Vu9F_RRr2dJRHv9neJzgAYlcc4v8m9nc51VCXZcwhXYMn8GwtfaJuVsBitUCJq8" alt="image-20230809192855023" />

图片攻击：运行**attr_test.py**，将对**input_img/test.jpg**进行各种图片攻击，攻击后的图片结果保存在**output_img**当中。

<img src="https://immich.lyh27.top/api/assets/c336ab46-7c7c-4141-b538-6457ee8d24d1/thumbnail?size=preview&key=25V6Vu9F_RRr2dJRHv9neJzgAYlcc4v8m9nc51VCXZcwhXYMn8GwtfaJuVsBitUCJq8" alt="image-20230809192718233" />

## 附录

项目参考于GitHub开源仓库 [blind_watermark](https://github.com/guofei9987/blind_watermark)

原作者[@guofei9987](https://github.com/guofei9987)
