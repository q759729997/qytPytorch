"""
    main_module - 图片数据增强，测试是将对应方法的@unittest.skip注释掉.

    Main members:

        # __main__ - 程序入口.
"""
import unittest
import sys

import torchvision

sys.path.insert(0, './')  # 定义搜索路径的优先顺序，序号从0开始，表示最大优先级

import qytPytorch  # noqa
print('qytPytorch module path :{}'.format(qytPytorch.__file__))  # 输出测试模块文件位置
from qytPytorch.utils.matplotlib_utils import show_image  # noqa
from qytPytorch.utils.matplotlib_utils import show_image_augmentation  # noqa


class TestImageAugmentation(unittest.TestCase):
    """ 图片数据增强.

    Main methods:
        test_show_image - 显示图片.
        test_RandomHorizontalFlip - 图片水平(左右)翻转.
        test_RandomVerticalFlip - 垂直（上下）翻转.
        test_RandomResizedCrop - 随机裁剪.
        test_ColorJitter_brightness - 亮度变化.
        test_ColorJitter_hue - 色调变化.
        test_ColorJitter_contrast - 对比度变化.
        test_Compose - 叠加多个图像增广方法.
    """
    img_file_name = './test/dataset/cv/imgs/catdog.jpg'

    @unittest.skip('debug')
    def test_show_image(self):
        """ 显示图片.
        """
        print('{} test_show_image {}'.format('-'*15, '-'*15))
        show_image(self.img_file_name)  # 直接弹出图片

    @unittest.skip('debug')
    def test_RandomHorizontalFlip(self):
        """ 图片水平(左右)翻转.
        """
        print('{} test_RandomHorizontalFlip {}'.format('-'*15, '-'*15))
        show_image_augmentation(self.img_file_name, augmentation_func=torchvision.transforms.RandomHorizontalFlip())  # 直接弹出图片

    @unittest.skip('debug')
    def test_RandomVerticalFlip(self):
        """ 垂直（上下）翻转.
        """
        print('{} test_RandomVerticalFlip {}'.format('-'*15, '-'*15))
        show_image_augmentation(self.img_file_name, augmentation_func=torchvision.transforms.RandomVerticalFlip())  # 直接弹出图片

    @unittest.skip('debug')
    def test_RandomResizedCrop(self):
        """ 随机裁剪.
        """
        print('{} test_RandomResizedCrop {}'.format('-'*15, '-'*15))
        augmentation_func = torchvision.transforms.RandomResizedCrop(200, scale=(0.1, 1), ratio=(0.5, 2))
        show_image_augmentation(self.img_file_name, augmentation_func=augmentation_func)  # 直接弹出图片

    @unittest.skip('debug')
    def test_ColorJitter_brightness(self):
        """ 亮度变化.
        """
        print('{} test_ColorJitter_brightness {}'.format('-'*15, '-'*15))
        augmentation_func = torchvision.transforms.ColorJitter(brightness=0.5)
        show_image_augmentation(self.img_file_name, augmentation_func=augmentation_func)  # 直接弹出图片

    @unittest.skip('debug')
    def test_ColorJitter_hue(self):
        """ 色调变化.
        """
        print('{} test_ColorJitter_hue {}'.format('-'*15, '-'*15))
        augmentation_func = torchvision.transforms.ColorJitter(hue=0.5)
        show_image_augmentation(self.img_file_name, augmentation_func=augmentation_func)  # 直接弹出图片

    @unittest.skip('debug')
    def test_ColorJitter_contrast(self):
        """ 对比度变化.
        """
        print('{} test_ColorJitter_contrast {}'.format('-'*15, '-'*15))
        augmentation_func = torchvision.transforms.ColorJitter(contrast=0.5)
        show_image_augmentation(self.img_file_name, augmentation_func=augmentation_func)  # 直接弹出图片

    # @unittest.skip('debug')
    def test_Compose(self):
        """ 叠加多个图像增广方法.
        """
        print('{} test_Compose {}'.format('-'*15, '-'*15))
        flip_aug = torchvision.transforms.RandomHorizontalFlip()
        shape_aug = torchvision.transforms.RandomResizedCrop(200, scale=(0.1, 1), ratio=(0.5, 2))
        color_aug = torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
        augmentation_func = torchvision.transforms.Compose([flip_aug, shape_aug, color_aug])
        show_image_augmentation(self.img_file_name, augmentation_func=augmentation_func)  # 直接弹出图片


if __name__ == "__main__":
    unittest.main()  # 运行当前源文件中的所有测试用例
