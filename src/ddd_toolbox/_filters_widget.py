import os
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QTabWidget,
                            QGroupBox, QHBoxLayout, QLabel, 
                            QComboBox, QCheckBox, QLineEdit, 
                            QPushButton, QFileDialog, QDoubleSpinBox
)
from qtpy.QtCore import Qt, QThread

import napari
from napari.utils import progress

import numpy as np

from skimage.filters import (gaussian, median, sobel, difference_of_gaussians)
from scipy.ndimage import (gaussian_laplace, minimum_filter, maximum_filter, 
                           uniform_filter)

class ImageFiltersWidget(QWidget):
    
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.filters = {
            "Gaussian": self.gaussian,
            "Median": self.median,
            "Sobel": self.sobel,
            "Laplacian of Gaussian": self.laplace,
            "Difference of Gaussians": self.difference_of_gaussians,
            "Minimum": self.minimum,
            "Maximum": self.maximum,
            "Mean": self.mean
        }

        self.init_ui()
        self.viewer.layers.events.inserted.connect(lambda e: self.update_info_label())
        self.viewer.layers.events.removed.connect(lambda e: self.update_info_label())
        self.viewer.layers.events.reordered.connect(lambda e: self.update_info_label())
    
    def init_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)
        self.create_filter_ui(layout)
    
    def create_filter_ui(self, layout):
        filters_group = QGroupBox("Filters")
        filters_layout = QVBoxLayout()
        filters_group.setLayout(filters_layout)
        layout.addWidget(filters_group)

        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel("Filter:"))
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(self.filters.keys())
        h_layout.addWidget(self.filter_combo)
        filters_layout.addLayout(h_layout)

        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel("Sigma 1:"))
        self.sigma1_spin = QDoubleSpinBox()
        self.sigma1_spin.setRange(0.0, 1000.0)
        self.sigma1_spin.setValue(1.0)
        h_layout.addWidget(self.sigma1_spin)
        filters_layout.addLayout(h_layout)

        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel("Sigma 2:"))
        self.sigma2_spin = QDoubleSpinBox()
        self.sigma2_spin.setRange(0.0, 1000.0)
        self.sigma2_spin.setValue(2.0)
        h_layout.addWidget(self.sigma2_spin)
        filters_layout.addLayout(h_layout)

        self.physical_units_checkbox = QCheckBox("Use Physical Units")
        self.physical_units_checkbox.stateChanged.connect(self.on_physical_units_update)
        filters_layout.addWidget(self.physical_units_checkbox)

        self.info_label = QLabel("---")
        filters_layout.addWidget(self.info_label)

        self.filter_combo.currentIndexChanged.connect(self.on_filter_update)
        self.sigma1_spin.valueChanged.connect(self.on_s1_update)
        self.sigma2_spin.valueChanged.connect(self.on_s2_update)

        self.apply_button = QPushButton("Apply Filter")
        filters_layout.addWidget(self.apply_button)
        self.apply_button.clicked.connect(self.apply_filter)
        self.on_filter_update()

    def on_filter_update(self):
        key = self.filter_combo.currentText()
        if key == "Difference of Gaussians":
            self.sigma1_spin.setEnabled(True)
            self.sigma2_spin.setEnabled(True)
        elif key == "Sobel":
            self.sigma1_spin.setEnabled(False)
            self.sigma2_spin.setEnabled(False)
        else:
            self.sigma1_spin.setEnabled(True)
            self.sigma2_spin.setEnabled(False)
        self.update_info_label()

    def on_s1_update(self):
        self.update_info_label()

    def on_s2_update(self):
        self.update_info_label()

    def on_physical_units_update(self):
        self.update_info_label()

    def update_info_label(self):
        s1 = self.process_pxl_kernel_size(self.sigma1_spin.value())
        s2 = self.process_pxl_kernel_size(self.sigma2_spin.value())
        if s1 is None or s2 is None:
            self.info_label.setText("---")
            return
        op = self.filter_combo.currentText()
        if op == "Difference of Gaussians":
            info = f"X: {s1[2]}, Y: {s1[1]}, Z: {s1[0]} | X: {s2[2]}, Y: {s2[1]}, Z: {s2[0]}"
        else:
            info = f"X: {s1[2]}, Y: {s1[1]}, Z: {s1[0]}"
        self.info_label.setText(info)

    def process_pxl_kernel_size(self, sigma):
        layer = self.viewer.layers.selection.active
        if layer is None:
            return None
        clb = np.array(layer.scale[-3:])
        if clb is None:
            clb = np.array([1.0, 1.0, 1.0])
        anisotropy = clb[2] / clb[0] # ZYX
        yx = max(1, int(sigma)) # input in pixels
        z  = max(1, int(np.ceil(sigma * anisotropy))) # input in pixels
        if self.physical_units_checkbox.isChecked(): # input in physical units, to be converted in pixels
            yx = max(1, int(np.ceil(sigma / clb[2])))
            z  = max(1, int(np.ceil(sigma / clb[0])))
        return np.array([z, yx, yx])
    
    def gaussian(self):
        layer = self.viewer.layers.selection.active
        if layer is None:
            return
        sigma = self.process_pxl_kernel_size(self.sigma1_spin.value())
        if sigma is None:
            return
        filtered = gaussian(layer.data, sigma=sigma)
        name = layer.name + f" Gaussian {self.sigma1_spin.value()}"
        if name in self.viewer.layers:
            self.viewer.layers[name].data = filtered
        else:
            self.viewer.add_image(filtered, name=name, scale=layer.scale, units=layer.units)

    def median(self):
        layer = self.viewer.layers.selection.active
        if layer is None:
            return
        sigma = self.process_pxl_kernel_size(self.sigma1_spin.value())
        if sigma is None:
            return
        footprint = np.ones(sigma, dtype=np.uint8)
        filtered = median(layer.data, footprint=footprint)
        name = layer.name + f" Median {self.sigma1_spin.value()}"
        if name in self.viewer.layers:
            self.viewer.layers[name].data = filtered
        else:
            self.viewer.add_image(filtered, name=name, scale=layer.scale, units=layer.units)

    def sobel(self):
        layer = self.viewer.layers.selection.active
        if layer is None:
            return
        filtered = sobel(layer.data)
        name = layer.name + " Sobel"
        if name in self.viewer.layers:
            self.viewer.layers[name].data = filtered
        else:
            self.viewer.add_image(filtered, name=name, scale=layer.scale, units=layer.units)

    def laplace(self):
        layer = self.viewer.layers.selection.active
        if layer is None:
            return
        sigma = self.process_pxl_kernel_size(self.sigma1_spin.value())
        if sigma is None:
            return
        filtered = gaussian_laplace(layer.data, sigma=sigma)
        name = layer.name + f" LoG {self.sigma1_spin.value()}"
        if name in self.viewer.layers:
            self.viewer.layers[name].data = filtered
        else:
            self.viewer.add_image(filtered, name=name, scale=layer.scale, units=layer.units)

    def difference_of_gaussians(self):
        layer = self.viewer.layers.selection.active
        if layer is None:
            return
        sigma1 = self.process_pxl_kernel_size(self.sigma1_spin.value())
        sigma2 = self.process_pxl_kernel_size(self.sigma2_spin.value())
        if sigma1 is None or sigma2 is None:
            return
        filtered = difference_of_gaussians(layer.data, low_sigma=sigma1, high_sigma=sigma2)
        name = layer.name + f" DoG {self.sigma1_spin.value()}-{self.sigma2_spin.value()}"
        if name in self.viewer.layers:
            self.viewer.layers[name].data = filtered
        else:
            self.viewer.add_image(filtered, name=name, scale=layer.scale, units=layer.units)

    def minimum(self):
        layer = self.viewer.layers.selection.active
        if layer is None:
            return
        sigma = self.process_pxl_kernel_size(self.sigma1_spin.value())
        if sigma is None:
            return
        filtered = minimum_filter(layer.data, size=sigma)
        name = layer.name + f" Minimum {self.sigma1_spin.value()}"
        if name in self.viewer.layers:
            self.viewer.layers[name].data = filtered
        else:
            self.viewer.add_image(filtered, name=name, scale=layer.scale, units=layer.units)

    def maximum(self):
        layer = self.viewer.layers.selection.active
        if layer is None:
            return
        sigma = self.process_pxl_kernel_size(self.sigma1_spin.value())
        if sigma is None:
            return
        filtered = maximum_filter(layer.data, size=sigma)
        name = layer.name + f" Maximum {self.sigma1_spin.value()}"
        if name in self.viewer.layers:
            self.viewer.layers[name].data = filtered
        else:
            self.viewer.add_image(filtered, name=name, scale=layer.scale, units=layer.units)

    def mean(self):
        layer = self.viewer.layers.selection.active
        if layer is None:
            return
        sigma = self.process_pxl_kernel_size(self.sigma1_spin.value())
        if sigma is None:
            return
        filtered = uniform_filter(layer.data, size=sigma)
        name = layer.name + f" Mean {self.sigma1_spin.value()}"
        if name in self.viewer.layers:
            self.viewer.layers[name].data = filtered
        else:
            self.viewer.add_image(filtered, name=name, scale=layer.scale, units=layer.units)
    
    def apply_filter(self):
        key = self.filter_combo.currentText()
        filter_func = self.filters.get(key, None)
        if filter_func is not None:
            filter_func()


def loose_launch():
    viewer = napari.Viewer()
    widget = ImageFiltersWidget(viewer=viewer)
    viewer.window.add_dock_widget(widget)

    napari.run()

if __name__ == "__main__":
    loose_launch()