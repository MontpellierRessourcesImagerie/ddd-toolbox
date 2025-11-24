import numpy as np
from skimage.util import invert


class ImageCalculator(object):


    def __init__(self, image1, image2):
        super().__init__()
        self.image1 = image1
        self.image2 = image2
        self.operation = 'add'
        self.result = None


    def run(self):
        if self.operation == "add":
            self.add()
        if self.operation == "subtract":
            self.subtract()
        if self.operation == "multiply":
            self.multiply()
        if self.operation == "divide":
            self.divide()


    def add(self):
        self.result = np.add(self.image1, self.image2)


    def subtract(self):
        self.result = np.subtract(self.image1, self.image2)


    def multiply(self):
        self.result = np.multiply(self.image1, self.image2)


    def divide(self):
        self.result = np.divide(self.image1, self.image2)



class ImageInfo(object):


    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.result = {}


    def run(self):
        self.result['name'] = [self.layer.name]
        self.result['ndim'] = [str(self.layer.ndim)]
        self.result['shape'] = [str(self.layer.data.shape)]
        self.result['dtype'] = [self.layer.dtype.name]
        self.result['scale'] = [str(self.layer.scale)]
        self.result['units'] = [str(self.layer.units)]
        self.result['uuid'] = [str(self.layer.unique_id)]
        self.result['min'] = [str(np.min(self.layer.data))]
        self.result['max'] = [str(np.max(self.layer.data))]
        self.result['path'] = [""]
        if self.layer.source.path:
            self.result['path'] = [self.layer.source.path]



class Invert(object):


    def __init__(self, image):
        super().__init__()
        self.image = image
        self.result = None


    def run(self):
        self.result = invert(self.image)



class ImageTypeConverter(object):


    def __init__(self, image):
        super().__init__()
        self.image = image
        self.result = None
        self.targetType = np.float32


    def run(self):
        self.result = self.image.astype(self.targetType)


    @classmethod
    def types(cls):
        return ['uint8', 'uint16', 'uint32', 'uint64', 'int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']