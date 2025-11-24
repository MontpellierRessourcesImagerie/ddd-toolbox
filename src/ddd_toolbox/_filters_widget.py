import os
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QTabWidget,
                            QGroupBox, QHBoxLayout, QLabel, 
                            QComboBox, QCheckBox, QLineEdit, 
                            QPushButton, QFileDialog, QDoubleSpinBox
)
from qtpy.QtCore import Qt, QThread

import napari
from napari.utils import progress
from napari.utils.notifications import show_error, show_warning

import numpy as np

from skimage.filters import (gaussian, median, sobel, difference_of_gaussians)
from scipy.ndimage import (gaussian_laplace, minimum_filter, maximum_filter, 
                           uniform_filter)
from skimage.restoration import rolling_ball

class ImageFiltersWidget(QWidget):
    
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer
        self.filters = {
            "Gaussian"               : (self.gaussian, "sigma"),
            "Laplacian of Gaussian"  : (self.laplace, "sigma"),
            "Difference of Gaussians": (self.difference_of_gaussians, "sigma"),
            "Median"                 : (self.median, "kernel"),
            "Sobel"                  : (self.sobel, ""),
            "Minimum"                : (self.minimum, "kernel"),
            "Maximum"                : (self.maximum, "kernel"),
            "Mean"                   : (self.mean, "kernel"),
            "Rolling Ball"           : (self.rolling_ball, "kernel")
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
        self.sigma1_spin.setSingleStep(0.1)
        h_layout.addWidget(self.sigma1_spin)
        filters_layout.addLayout(h_layout)

        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel("Sigma 2:"))
        self.sigma2_spin = QDoubleSpinBox()
        self.sigma2_spin.setRange(0.0, 1000.0)
        self.sigma2_spin.setValue(2.0)
        self.sigma2_spin.setSingleStep(0.1)
        h_layout.addWidget(self.sigma2_spin)
        filters_layout.addLayout(h_layout)

        self.physical_units_checkbox = QCheckBox("Use physical length")
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
        s1 = self.process_pxl_size(self.sigma1_spin.value())
        s2 = self.process_pxl_size(self.sigma2_spin.value())
        if s1 is None or s2 is None:
            self.info_label.setText("---")
            return
        op = self.filter_combo.currentText()
        _, category = self.filters[op]
        if category == "kernel":
            s1 = self.as_kernel_radii(s1)
            s2 = self.as_kernel_radii(s2)
        if op == "Difference of Gaussians":
            info = f"X: {round(s1[2], 2)}, Y: {round(s1[1], 2)}, Z: {round(s1[0], 2)} | X: {round(s2[2], 2)}, Y: {round(s2[1], 2)}, Z: {round(s2[0], 2)}"
        else:
            info = f"X: {round(s1[2], 2)}, Y: {round(s1[1], 2)}, Z: {round(s1[0], 2)}"
        self.info_label.setText(info)

    def process_pxl_size(self, sigma):
        layer = self.viewer.layers.selection.active
        if layer is None or not hasattr(layer, 'colormap'):
            return None
        clb = np.array(layer.scale[-3:])
        if clb is None:
            clb = np.array([1.0, 1.0, 1.0])
        anisotropy = clb[2] / clb[0] # ZYX
        yx = sigma # input in pixels
        z  = sigma * anisotropy # input in pixels
        if self.physical_units_checkbox.isChecked(): # input in physical units, to be converted in pixels
            yx = sigma / clb[2]
            z  = sigma / clb[0]
        return np.array([z, yx, yx])
    
    def as_kernel_radii(self, sigmas):
        return [max(1, int(np.ceil(s))) for s in sigmas]
    
    def gaussian(self, data, s1, s2):
        data = data.astype(np.float32)
        filtered = np.zeros_like(data)
        for i in range(data.shape[0]):
            filtered[i] = gaussian(data[i], sigma=s1)
        return filtered

    def median(self, data, s1, s2):
        s1 = self.as_kernel_radii(s1)
        filtered = np.zeros_like(data)
        footprint = np.ones(2*np.array(s1)+1, dtype=np.uint8)
        for i in range(data.shape[0]):
            filtered[i] = median(data[i], footprint=footprint)
        return filtered

    def rolling_ball(self, data, s1, s2):
        s1 = self.as_kernel_radii(s1)
        filtered = np.zeros_like(data)
        footprint = np.ones(2*np.array(s1)+1, dtype=np.uint8)
        for i in range(data.shape[0]):
            filtered[i] = rolling_ball(data[i], kernel=footprint)
        return filtered

    def sobel(self, data, s1, s2):
        data = data.astype(np.float32)
        filtered = np.zeros_like(data)
        for i in range(data.shape[0]):
            filtered[i] = sobel(data[i])
        return filtered

    def laplace(self, data, s1, s2):
        data = data.astype(np.float32)
        filtered = np.zeros_like(data)
        for i in range(data.shape[0]):
            filtered[i] = gaussian_laplace(data[i], sigma=s1)
        return filtered

    def difference_of_gaussians(self, data, s1, s2):
        data = data.astype(np.float32)
        filtered = np.zeros_like(data)
        for i in range(data.shape[0]):
            filtered[i] = difference_of_gaussians(data[i], low_sigma=s1, high_sigma=s2)
        return filtered

    def minimum(self, data, s1, s2):
        s1 = self.as_kernel_radii(s1)
        filtered = np.zeros_like(data)
        for i in range(data.shape[0]):
            filtered[i] = minimum_filter(data[i], size=2*np.array(s1)+1)
        return filtered

    def maximum(self, data, s1, s2):
        s1 = self.as_kernel_radii(s1)
        filtered = np.zeros_like(data)
        for i in range(data.shape[0]):
            filtered[i] = maximum_filter(data[i], size=2*np.array(s1)+1)
        return filtered

    def mean(self, data, s1, s2):
        s1 = self.as_kernel_radii(s1)
        filtered = np.zeros_like(data)
        for i in range(data.shape[0]):
            filtered[i] = uniform_filter(data[i], size=2*np.array(s1)+1)
        return filtered
    
    def run_filter(self, filter_name):
        layer = self.viewer.layers.selection.active
        if layer is None or not hasattr(layer, 'colormap'):
            show_warning("No active image layer selected.")
            return
        s1 = self.process_pxl_size(self.sigma1_spin.value())
        s2 = self.process_pxl_size(self.sigma2_spin.value())
        if (s1 is None) or (s2 is None):
            show_error("Impossible to extract sigma values.")
            return
        filter_func, _ = self.filters.get(filter_name, (None, None))
        if filter_func is None:
            show_error(f"Filter {filter_name} not found.")
            return
        data = layer.data if layer.ndim == 4 else layer.data[np.newaxis, ...]
        filtered = filter_func(data, s1, s2)
        filtered = filtered if layer.ndim == 4 else filtered[0]
        name = layer.name + f" {filter_name}"
        if self.sigma1_spin.isEnabled():
            name += f" {round(self.sigma1_spin.value(), 2)}"
        if self.sigma2_spin.isEnabled():
            name += f"-{round(self.sigma2_spin.value(), 2)}"
        if name in self.viewer.layers:
            self.viewer.layers[name].data = filtered
        else:
            self.viewer.add_image(filtered, name=name, scale=layer.scale, units=layer.units)
    
    def apply_filter(self):
        key = self.filter_combo.currentText()
        if key not in self.filters:
            show_error("Filter not recognized.")
            return
        self.run_filter(key)


def loose_launch():
    viewer = napari.Viewer()
    widget = ImageFiltersWidget(viewer=viewer)
    viewer.window.add_dock_widget(widget)

    napari.run()

if __name__ == "__main__":
    loose_launch()