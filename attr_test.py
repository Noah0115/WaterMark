# coding=utf-8
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