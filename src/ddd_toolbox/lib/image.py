import numpy as np
from skimage.util import invert


class ImageCalculator(object):


    def __init__(self, image1, image2):
        self.type = image1.dtype
        self.image1 = image1.astype(np.float32)
        self.image2 = image2.astype(np.float32)
        self.operation = 'add'
        self.result = None
        self.operations = {
            'add': self.add, 
            'subtract': self.subtract, 
            'multiply': self.multiply, 
            'divide': self.divide,
            'difference': self.difference
        }

    def clamp(self, result):
        allowed_min = np.iinfo(self.type).min
        allowed_max = np.iinfo(self.type).max
        result = np.clip(result, allowed_min, allowed_max)
        return result.astype(self.type)

    def run(self):
        op = self.operations.get(self.operation, None)
        if op is None:
            print("Operation not recognized")
            return
        res = op()
        self.result = self.clamp(res)

    def difference(self):
        return np.abs(self.image1 - self.image2)


    def add(self):
        return np.add(self.image1, self.image2)


    def subtract(self):
        return np.subtract(self.image1, self.image2)


    def multiply(self):
        return np.multiply(self.image1, self.image2)


    def divide(self):
        return np.divide(self.image1, self.image2)



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