"""
    module(file_utils) - 文件处理工具类.

    Main members:

        # extract_tarfile - 解压tar或tar.gz文件.
"""
import os
import tarfile


def extract_tarfile(tarfile_name, output_path, target_file_name='target'):
    """ 解压tar或tar.gz文件.

        @params:
            tarfile_name - tar或tar.gz文件.
            output_path - 输出路径.
            target_file_name - 输出模板文件名称.

        @return:
            On success - 解压信息.
            On failure - 错误信息.
    """
    if os.path.exists(os.path.join(output_path, target_file_name)):
        return '{} already exist'.format(target_file_name)
    else:
        with tarfile.open(tarfile_name, mode='r') as fr:
            fr.extractall(output_path)
            return 'success'
