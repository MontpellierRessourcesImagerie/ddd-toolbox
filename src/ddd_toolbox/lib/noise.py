from time import time_ns
import numpy as np
from abc import abstractmethod
from skimage.util import random_noise

class AddNoise(object):
    """
    Abstract Class. Add noise to the image. The subclass specifies what kind of noise.
    """

    def __init__(self, input_image):
        self.image = input_image
        self.result = None
        self.isConvertBack = True
        self.clip = True


    @abstractmethod
    def run(self):
        raise Exception("Abstract method run of class Filter called!")


    def convertBack(self):
        if np.min(self.image) < 0:
            self.result = (self.result + 1) / 2.0
        if np.issubdtype(self.image.dtype, np.integer):
            min_value = np.iinfo(self.image.dtype).min
            max_value = np.iinfo(self.image.dtype).max
        else:
            min_value = np.finfo(self.image.dtype).min
            max_value = np.finfo(self.image.dtype).max
        self.result = self.result * (max_value - min_value) + min_value
        self.result = self.result.astype("uint16")



class AddGaussianNoise(AddNoise):
    """Add Gaussian noise to the image."""


    def __init__(self, input_image):
        super().__init__(input_image)
        self.mean = 0
        self.stdDev = 0.1


    def run(self):
        seed = time_ns()
        variation = self.stdDev * self.stdDev
        self.result = random_noise(self.image,
                                   mode='gaussian',
                                   rng=seed,
                                   mean=0,
                                   var=variation
                                   )
        if self.isConvertBack:
            self.convertBack()



class AddPoissonNoise(AddNoise):


    def __init__(self, input_image):
        super().__init__(input_image)
        self.mean = 0
        self.stdDev = 0.1


    def run(self):
        seed = time_ns()
        self.result = random_noise(self.image,
                                   mode='poisson',
                                   rng=seed
                                   )
        if self.isConvertBack:
            self.convertBack()