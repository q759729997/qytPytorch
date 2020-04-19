"""
    module(matplotlib_utils) - matplotlib绘图工具类.

    Main members:

        # set_figsize - 设置图片显示尺寸.
        # show_x_y_axis - 显示x、y坐标系.
        # show_image - 显示图片.
        # show_image_augmentation - 显示图片增强.
"""
import pylab
# from IPython import display
from PIL import Image
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


def show_x_y_axis(data_x, data_y, label_x='x axis', label_y='y axis', title='matplotlib show', annotates=list()):
    """ 显示x、y坐标系.

        @params:
            data_x - x轴数据.
            data_y - y轴数据.
            label_x - x轴标签.
            label_y - y轴标签.
            title - 标题.
            annotates - 注解[{'text': 文本, 'xy': 函数曲线坐标, 'xytext': 文本所在坐标, 'arrowstyle': 箭头类型}].
    """
    plt.title(title)
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.plot(data_x, data_y)
    for annotate_item in annotates:
        arrowstyle = annotate_item.get('arrowstyle', '->')
        plt.annotate(annotate_item['text'], xy=annotate_item['xy'], xytext=annotate_item['xytext'], arrowprops=dict(arrowstyle=arrowstyle))
    plt.show()


def show_image(img_file_name):
    """ 显示图片.

        @params:
            img_file_name - 图片文件路径.
    """
    img_obj = Image.open(img_file_name)
    plt.imshow(img_obj)
    pylab.show()  # 直接利用plt.imshow()发现居然不能显示图片


def show_image_augmentation(img_file_name, augmentation_func, num_rows=2, num_cols=2, scale=2):
    """ 显示图片增强.

        @params:
            img_file_name - 图片文件路径.
            augmentation_func - 图片增强方法，torchvision.transforms中的方法.
            num_rows - 行数.
            num_cols - 列数.
            scale - 比例.
    """
    img_obj = Image.open(img_file_name)
    # 对输入图像img多次运行图像增广方法augmentation_func并展示所有的结果，行数和列数控制运行次数
    imgs = [augmentation_func(img_obj) for _ in range(num_rows * num_cols)]
    # figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols)  # , figsize=figsize
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i][j].imshow(imgs[i * num_cols + j])
            axes[i][j].axes.get_xaxis().set_visible(False)
            axes[i][j].axes.get_yaxis().set_visible(False)
    pylab.show()  # 直接利用plt.imshow()发现居然不能显示图片
