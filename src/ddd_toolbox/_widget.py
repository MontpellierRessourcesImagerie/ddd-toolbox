"""
This module contains 3D image analysis widgets which provide a gui
for common python image analysis operations, for example from skimage
or scipy.
"""

import numpy as np
from napari.qt.threading import create_worker
from autooptions import Options

from ddd_toolbox.simple_widget import SimpleWidget
from ddd_toolbox.lib.qtutil import TableView
from ddd_toolbox.lib.filter import ConvolutionFilter
from ddd_toolbox.lib.transform import FFT, InverseFFT
from ddd_toolbox.lib.image import ImageCalculator, ImageInfo, Invert
from ddd_toolbox.lib.noise import AddGaussianNoise, AddPoissonNoise
from ddd_toolbox.lib.measure import TableTool
from ddd_toolbox.lib.tools import RenameListElements
from ddd_toolbox.lib.image import ImageTypeConverter

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import napari

class ConvolutionWidget(SimpleWidget):


    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__(viewer)


    @classmethod
    def getOptions(cls):
        options = Options("3D Toolbox", "Convolution Filter")
        options.addImage()
        options.addImage(name='kernel')
        options.addChoice("mode", value='same', choices=('same', 'valid', 'full'))
        options.addChoice("method", value='auto', choices=('auto', 'direct', 'fft'))
        options.load()
        return options


    def apply(self):
        self.imageLayer = self.widget.getImageLayer('image')
        kernelLayer = self.widget.getImageLayer('kernel')
        self.operation = ConvolutionFilter(self.imageLayer.data, kernelLayer.data)
        self.operation.mode = self.options.value('mode')
        self.operation.method = self.options.value('method')
        worker = create_worker(self.operation.run,
                               _progress={'desc': 'Applying convolution filter...'}
                               )
        worker.finished.connect(self.displayResult)
        worker.start()


    def displayResult(self):
        name = self.imageLayer.name + " convolution"
        self.displayImage(name)
        


class FFTWidget(SimpleWidget):
    

    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__(viewer)


    def getOptions(self):
        options = Options("3D Toolbox", "FFT")
        options.addImage()
        options.load()
        return options


    def apply(self):
        self.imageLayer = self.widget.getImageLayer('image')
        self.operation = FFT(self.imageLayer.data)
        worker = create_worker(self.operation.run,
                               _progress={'desc': 'Applying FFT...'}
                               )
        worker.finished.connect(self.displayResult)
        worker.start()


    def displayResult(self):
        name = self.imageLayer.name + " FFT"
        powerSpectrum = self.operation.getPowerSpectrum()
        layer = self.viewer.add_image(
            powerSpectrum,
            name=name,
            scale=self.imageLayer.scale,
            units=self.imageLayer.units,
            blending='additive',
            colormap='inferno'
        )
        layer.metadata['fft'] = self.operation.result



class InverseFFTWidget(SimpleWidget):


    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__(viewer)


    def getOptions(self):
        options = Options("3D Toolbox", "Inverse FFT")
        options.addFFT()
        options.load()
        return options


    def apply(self):
        self.imageLayer = self.widget.getImageLayer('image')
        self.operation = InverseFFT(self.imageLayer.metadata['fft'])
        if 'ifftshift' in self.imageLayer.metadata.keys() and self.imageLayer.metadata['ifftshift']:
            self.operation.ifftshift = True
        worker = create_worker(self.operation.run,
                               _progress={'desc': 'Applying Inverse FFT...'}
                               )
        worker.finished.connect(self.displayResult)
        worker.start()


    def displayResult(self):
        name = self.imageLayer.name + " inverse FFT"
        self.displayImage(name)



class ImageCalculatorWidget(SimpleWidget):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__(viewer)
        self.isFFT = False


    def getOptions(self):
        options = Options("3D Toolbox", "calculator")
        options.addImage(name="image 1")
        options.addImage(name="image 2")
        options.addChoice("operation", value='multiply', choices=('add', 'subtract', 'multiply', 'divide', 'difference', 'or', 'and'))
        options.load()
        return options


    def apply(self):
        imageLayer1 = self.widget.getImageLayer('image 1')
        imageLayer2 = self.widget.getImageLayer('image 2')
        self.imageLayer = imageLayer1
        operationOption = self.options.value('operation')
        image1 = imageLayer1.data
        image2 = imageLayer2.data
        self.isFFT = False
        if 'fft' in imageLayer1.metadata.keys() and 'fft' in imageLayer2.metadata.keys():
            image1 = imageLayer1.metadata['fft']
            image2 = imageLayer2.metadata['fft']
            self.isFFT = True
        self.operation = ImageCalculator(image1, image2)
        self.operation.operation = operationOption
        worker = create_worker(self.operation.run,
                               _progress={'desc': 'Applying Image Calculator...'}
                               )
        worker.finished.connect(self.displayResult)
        worker.start()


    def displayResult(self):
        name = self.imageLayer.name + " calc"
        if not self.isFFT:
            self.displayImage(name)
            return
        else:
            fft = FFT(None)
            fft.result = self.operation.result
            powerSpectrum = fft.getPowerSpectrum()
            layer = self.viewer.add_image(
                powerSpectrum,
                name=name,
                scale=self.imageLayer.scale,
                units=self.imageLayer.units,
                blending='additive',
                colormap='inferno'
            )
            layer.metadata['fft'] = self.operation.result
            layer.metadata['ifftshift'] = True



class AddGaussianNoiseWidget(SimpleWidget):


    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__(viewer)


    def getOptions(self):
        options = Options("3D Toolbox", "add gaussian noise")
        options.addImage()
        options.addFloat("mean", value=0.0)
        options.addFloat("stdDev", value=0.1)
        options.load()
        return options


    def apply(self):
        self.imageLayer = self.widget.getImageLayer("image")
        self.operation = AddGaussianNoise(self.imageLayer.data)
        self.operation.mean = self.options.value('mean')
        self.operation.stdDev = self.options.value('stdDev')
        worker = create_worker(self.operation.run,
                               _progress={'desc': 'Adding Gaussian Noise...'}
                               )
        worker.finished.connect(self.displayResult)
        worker.start()


    def displayResult(self):
        name = self.imageLayer.name + " gaussian noise"
        self.displayImage(name)



class AddPoissonNoiseWidget(SimpleWidget):


    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__(viewer)


    def getOptions(self):
        options = Options("3D Toolbox", "add poisson noise")
        options.addImage()
        options.load()
        return options


    def apply(self):
        self.imageLayer = self.widget.getImageLayer("image")
        self.operation = AddPoissonNoise(self.imageLayer.data)
        worker = create_worker(self.operation.run,
                               _progress={'desc': 'Adding Poisson Noise...'}
                               )
        worker.finished.connect(self.displayResult)
        worker.start()


    def displayResult(self):
        name = self.imageLayer.name + " poisson noise"
        self.displayImage(name)



class ImageInfoWidget(SimpleWidget):


    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__(viewer)
        self.data = {}
        self.table = TableView(self.data)
        self.tableDockWidget = None


    def getOptions(self):
        options = Options("3D Toolbox", "image info")
        options.addImage()
        options.load()
        return options


    def apply(self):
        self.imageLayer = self.widget.getImageLayer("image")
        self.operation = ImageInfo(self.imageLayer)
        worker = create_worker(self.operation.run,
                               _progress={'desc': 'Reading Image Info...'}
                               )
        worker.finished.connect(self.displayResult)
        worker.start()


    def displayResult(self):
        name = "Image Info"
        area = "right"
        if "Image Info" in self.viewer.window.dock_widgets.keys():
            self.viewer.window.remove_dock_widget(self.viewer.window.dock_widgets[name])
        TableTool.addTableAToB(self.operation.result, self.data)
        self.operation.result = {}
        self.table = TableView(self.data)
        self.table.resetAction.triggered.connect(self.resetTable)
        self.table.deleteAction.triggered.connect(self.deleteSelectedRows)
        self.tableDockWidget = self.viewer.window.add_dock_widget(self.table,
                                           area='right', name=name, tabify=True)


    def resetTable(self):
        for key in self.data.keys():
            self.data[key] = []
        self.displayResult()


    def deleteSelectedRows(self):
        indexes = self.table.selectedIndexes()
        indexes = [index.row() for index in indexes]
        indexes = list(set(indexes))
        print("indexes", indexes)
        print("type", type(indexes))
        for key in self.data.keys():
            data = np.array(self.data[key])
            data = np.delete(data, indexes)
            self.data[key] =  data.tolist()
        self.displayResult()



class InvertImageWidget(SimpleWidget):


    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__(viewer)


    def getOptions(self):
        options = Options("3D Toolbox", "invert image")
        options.addImage()
        options.load()
        return options


    def apply(self):
        self.imageLayer = self.widget.getImageLayer("image")
        self.operation = Invert(self.imageLayer.data)
        worker = create_worker(self.operation.run,
                               _progress={'desc': 'Inverting Image...'}
                               )
        worker.finished.connect(self.displayResult)
        worker.start()


    def displayResult(self):
        name = self.imageLayer.name + " invert"
        self.displayImage(name)



class RenameLayersWidget(SimpleWidget):


    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__(viewer)


    def getOptions(self):
        options = Options("3D Toolbox", "rename layers")
        options.addStr("old text", value='_')
        options.addStr("new text", value='-')
        options.addBool("flip", value=False)
        options.load()
        return options


    def apply(self):
        names = [layer.name for layer in self.viewer.layers]
        self.operation = RenameListElements(names)
        self.operation.subString = self.options.value('old text')
        self.operation.newSubString = self.options.value('new text')
        self.operation.isFlip = self.options.value('flip')
        worker = create_worker(self.operation.run,
                               _progress={'desc': 'Renaming layers...'}
                               )
        worker.finished.connect(self.displayResult)
        worker.start()


    def displayResult(self):
        newNames = self.operation.result
        for name, layer in zip(newNames, self.viewer.layers):
            layer.name = name



class ConvertImageTypeWidget(SimpleWidget):


    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__(viewer)


    def getOptions(self):
        options = Options("3D Toolbox", "convert image type")
        options.addImage()
        options.addChoice("type", value='float32', choices=ImageTypeConverter.types())
        options.load()
        return options


    def apply(self):
        self.imageLayer = self.widget.getImageLayer("image")
        self.operation = ImageTypeConverter(self.imageLayer.data)
        self.operation.targetType = self.options.value('type')
        worker = create_worker(self.operation.run,
                               _progress={'desc': 'Converting Image Type...'}
                               )
        worker.finished.connect(self.displayResult)
        worker.start()


    def displayResult(self):
        name = self.imageLayer.name + " convert"
        self.displayImage(name)


