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
