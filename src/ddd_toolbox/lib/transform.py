import numpy as np
from abc import abstractmethod
from scipy.fft import fftn
from scipy.fft import ifftn
from scipy.fft import fftshift



class Transform(object):
    """
    Abstract Filter class. A filter has an input image and a result image. It also has a mode telling how
    the missing information at the edges of the image is handled.
    """

    def __init__(self, input_image):
        self.image = input_image
        self.result = None


    @abstractmethod
    def run(self):
        raise Exception("Abstract method run of class Filter called!")


    
class FFT(Transform):
    """
    Calculate the discrete fourier transform of the input image. The result is a complex valued image.
    """


    def __init__(self, image):
        super().__init__(image)


    def run(self):
        self.result = fftn(self.image)


    def getPowerSpectrum(self):
        shifted = fftshift(self.result)
        powerSpectrum = np.abs(shifted) ** 2
        return powerSpectrum



class InverseFFT(Transform):
    """
    Calculate the inverse transform of the discrete fourier transform. The result is a real valued image.
    """


    def __init__(self, image):
        super().__init__(image)


    def run(self):
        self.result = ifftn(self.image).real