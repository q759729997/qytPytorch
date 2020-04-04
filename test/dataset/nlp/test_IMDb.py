"""
    main_module - IMDb情感分类常用函数，测试是将对应方法的@unittest.skip注释掉.

    Main members:

        # __main__ - 程序入口.
"""
import os
import unittest
import sys

sys.path.insert(0, './')  # 定义搜索路径的优先顺序，序号从0开始，表示最大优先级

import qytPytorch  # noqa
print('qytPytorch module path :{}'.format(qytPytorch.__file__))  # 输出测试模块文件位置
from qytPytorch.utils.file_units import extract_tarfile  # noqa


class TestIMDb(unittest.TestCase):
    """ IMDb情感分类常用函数.

    Main methods:
        test_get_dataset - 获取数据集.
    """

    # @unittest.skip('debug')
    def test_get_dataset(self):
        """ 获取数据集.
        下载链接(压缩包越80MB)： http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
        """
        print('{} test_get_dataset {}'.format('-'*15, '-'*15))
        # 数据解压
        data_path = './data'
        tarfile_name = os.path.join(data_path, 'aclImdb_v1.tar.gz')
        print(extract_tarfile(tarfile_name, output_path=data_path, target_file_name='aclImdb'))


if __name__ == "__main__":
    unittest.main()  # 运行当前源文件中的所有测试用例
