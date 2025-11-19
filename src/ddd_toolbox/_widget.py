"""
This module contains 3D image analysis widgets which provide a gui
for common python image analysis operations, for example from skimage
or scipy.
"""
from abc import abstractmethod

from PyQt5.QtWidgets import QVBoxLayout
from qtpy.QtWidgets import QWidget
from napari.qt.threading import create_worker
from autooptions import Options
from autooptions import OptionsWidget
from ddd_toolbox.lib.filter import ConvolutionFilter
from ddd_toolbox.lib.transform import FFT, InverseFFT

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import napari



class SimpleWidget(QWidget):


    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.options = self.getOptions()
        self.widget = None
        self.operation = None
        self.imageLayer = None
        self.createLayout()


    def createLayout(self):
        self.widget = OptionsWidget(self.viewer, self.options)
        self.widget.addApplyButton(self.apply)
        layout = QVBoxLayout()
        layout.addWidget(self.widget)
        self.setLayout(layout)


    def displayImage(self, name):
        self.viewer.add_image(
            self.operation.result,
            name=name,
            scale=self.imageLayer.scale,
            units=self.imageLayer.units,
            blending='additive'
        )


    @abstractmethod
    def getOptions(self):
        raise Exception("Abstract method getOptions of class SimpleWidget called!")


    @abstractmethod
    def apply(self):
        raise Exception("Abstract method apply of class SimpleWidget called!")


    @abstractmethod
    def displayResult(self):
        raise Exception("Abstract method displayResult of class SimpleWidget called!")



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
        worker = create_worker(self.operation.run,
                               _progress={'desc': 'Applying Inverse FFT...'}
                               )
        worker.finished.connect(self.displayResult)
        worker.start()


    def displayResult(self):
        name = self.imageLayer.name + " inverse FFT"
        self.displayImage(name)