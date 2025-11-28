from abc import abstractmethod

from PyQt5.QtWidgets import QVBoxLayout
from qtpy.QtWidgets import QWidget
from autooptions import OptionsWidget

from typing import TYPE_CHECKING


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
        return self.viewer.add_image(
            self.operation.result,
            name=name,
            scale=self.imageLayer.scale,
            units=self.imageLayer.units,
            blending='additive',
            colormap=self.imageLayer.colormap
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
