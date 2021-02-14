import cv2


class Frame:
    """Utility for easier color format conversions.

    :param image: OpenCV image as numpy array.
    :param color_format: Name of input color format or None.
         If str, the input must use the format that is used in OpenCV's cvtColor code parameter.
         For example, if an image is bgr then input 'BGR' as seen in the cvtColor codes:
        [cv2.COLOR_BGR2GRAY, COLOR_Luv2BGR].
        If None, the color format will be assumed from shape of the image.
        The only possible outcomes of this assumption are: ['GRAY', 'BGR', 'BGRA'].

    :ivar image: input image with possible color format conversions applied
    :ivar color_format: str containing the current color format of image attribute.
    """
    def __init__(self, image, color_format=None):
        # 图片的np array
        self.image = image

        # 没有给定颜色方案
        if color_format is None:
            # 根据维数从3个选项中猜测一个
            self.color_format = self._guess_color_format()
        # 帧的颜色方案：self.color_format
        else:
            self.color_format = color_format

    def _guess_color_format(self):
        # 2维：GRAY
        if len(self.image.shape) == 2:
            return 'GRAY'

        # 3维且第3维长度是3：BGR
        elif self.image.shape[2] == 3:
            return 'BGR'

        # 3维且第3维长度是4：BGRA
        elif self.image.shape[2] == 4:
            return 'BGRA'

        # 无法处理的情况
        else:
            raise ValueError(f'Unexpected frame image shape: {self.image.shape}')

    @staticmethod
    def _lookup_color_conversion(from_format, to_format):
        # 例如：cv2.COLOR_BGR2RGB
        return getattr(cv2, f'COLOR_{from_format}2{to_format}')

    def cvt_color(self, to_format):
        # 转换颜色方案

        # 当前方案与目标方案不一致
        if not self.color_format == to_format:
            # 查找cv2中的颜色转换属性
            color_conversion = self._lookup_color_conversion(from_format=self.color_format,
                                                             to_format=to_format)

            # 实施转换
            return cv2.cvtColor(self.image, color_conversion)
        # 一致，直接返回
        else:
            return self.image

    @property
    def gray_image(self):
        # 返回GRAY图
        return self.cvt_color('GRAY')

    @property
    def bgr_image(self):
        # 返回BGR图
        return self.cvt_color('BGR')

    @property
    def bgra_image(self):
        # 返回BGRA图
        return self.cvt_color('BGRA')
