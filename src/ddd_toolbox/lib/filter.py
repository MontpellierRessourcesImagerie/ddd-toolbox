from abc import abstractmethod

import numpy as np
from scipy.signal import convolve


class Filter(object):
    """
    Abstract Filter class. A filter has an input image and a result image. It also has a mode telling how
    the missing information at the edges of the image is handled.
    """

    def __init__(self, input_image):
        self.image = input_image
        self.mode='reflect'
        self.result = None


    @abstractmethod
    def run(self):
        raise Exception("Abstract method run of class Filter called!")



class ConvolutionFilter(Filter):
    """
    A convolution filter runs a convolution of the input image with the kernel. According to the method selected,
    the convolution is calculated in the spatial or in the fourier domain.
    """


    def __init__(self, input_image, kernel):
        super().__init__(input_image)
        self.kernel = kernel
        self.mode = 'same'              # Convolution mode {‘full’, ‘valid’, ‘same’}, not border mode as in superclass
        self.method = 'auto'            # Calculation mode fft or spatial domain  {‘auto’, ‘direct’, ‘fft’}
        self.isConvertBack = True


    def run(self):
        self.result = convolve(self.image, self.kernel, mode=self.mode, method=self.method)
        if self.isConvertBack:
            self.result = self.result.astype(np.uint16)


