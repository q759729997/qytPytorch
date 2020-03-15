"""
    module(matplotlib_utils) - matplotlib绘图工具类.

    Main members:

        # set_figsize - 设置图片显示尺寸.
        # show_x_y_axis - 显示x、y坐标系.
"""
# from IPython import display
from matplotlib import pyplot as plt


# def use_svg_display():
#     # 用矢量图显示
#     display.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):
    """ 设置图片显示尺寸.

        @params:
            figsize - 图片显示尺寸,元组形式.
    """
    # use_svg_display()
    plt.rcParams['figure.figsize'] = figsize


def show_x_y_axis(data_x, data_y, label_x='x axis', label_y='y axis', title='matplotlib show'):
    """ 显示x、y坐标系.

        @params:
            data_x - x轴数据.
            data_y - y轴数据.
            label_x - x轴标签.
            label_y - y轴标签.
            title - 标题.
    """
    plt.title(title)
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.plot(data_x, data_y)
    plt.show()
