import os
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QTabWidget,
                            QGroupBox, QHBoxLayout, QLabel, 
                            QComboBox, QCheckBox, QLineEdit, 
                            QPushButton, QFileDialog, QDoubleSpinBox
)
from qtpy.QtCore import Qt, QThread

import napari
from napari.utils import progress
from napari.utils import Colormap

import numpy as np

from skimage.filters import threshold_li, threshold_mean, threshold_otsu, threshold_triangle, threshold_yen
from skimage.measure import label
from scipy.ndimage import distance_transform_edt, grey_closing, grey_opening, white_tophat, black_tophat

NEUTRAL = "--------"

class MaskUtils(QWidget):
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.threshold_methods = {
            "Manual": None,
            "Li" : threshold_li,
            "Mean" : threshold_mean,
            "Otsu" : threshold_otsu,
            "Triangle": threshold_triangle,
            "Yen" : threshold_yen
        }
        self.kernels = {
            "Ball": self.ball,
            "Cube": self.cube
        }
        self.init_ui()

        self.refresh_layer_names()
        self.viewer.layers.events.inserted.connect(lambda e: self.refresh_layer_names())
        self.viewer.layers.events.removed.connect(lambda e: self.refresh_layer_names())
        self.viewer.layers.events.reordered.connect(lambda e: self.refresh_layer_names())

    def init_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)
        self.init_threshold_ui(layout)
        self.create_morphology_ui(layout)
        self.init_labeling_ui(layout)
    
    def init_threshold_ui(self, layout):
        threshold_group = QGroupBox("Thresholding")
        threshold_layout = QVBoxLayout()
        threshold_group.setLayout(threshold_layout)

        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel("Bounds:"))
        self.lower_spin = QDoubleSpinBox()
        self.lower_spin.setRange(-1e10, 1e10)
        self.lower_spin.setValue(0.0)
        self.lower_spin.setPrefix("Low ")
        h_layout.addWidget(self.lower_spin)

        self.upper_spin = QDoubleSpinBox()
        self.upper_spin.setRange(-1e10, 1e10)
        self.upper_spin.setValue(1.0)
        self.upper_spin.setPrefix("High ")
        h_layout.addWidget(self.upper_spin)

        threshold_layout.addLayout(h_layout)

        self.dark_bg_check = QCheckBox("Dark Background")
        self.dark_bg_check.setChecked(True)
        self.dark_bg_check.stateChanged.connect(self.on_dark_bg_update)
        threshold_layout.addWidget(self.dark_bg_check)

        self.method_combobox = QComboBox()
        self.method_combobox.addItems(self.threshold_methods.keys())
        self.method_combobox.currentIndexChanged.connect(self.on_method_update)
        threshold_layout.addWidget(self.method_combobox)

        h_layout = QHBoxLayout()
        self.individual_frames_check = QCheckBox("Individual frames")
        self.individual_frames_check.stateChanged.connect(self.on_method_update)
        h_layout.addWidget(self.individual_frames_check)
        self.apply_button = QPushButton("Threshold")
        self.apply_button.clicked.connect(self.apply_threshold)
        h_layout.addWidget(self.apply_button)
        threshold_layout.addLayout(h_layout)

        layout.addWidget(threshold_group)

    def create_morphology_ui(self, layout):
        transforms_group = QGroupBox("Morphology")
        transforms_layout = QVBoxLayout()
        transforms_group.setLayout(transforms_layout)
        layout.addWidget(transforms_group)

        self.kernels_combo_box = QComboBox()
        self.kernels_combo_box.addItems(self.kernels.keys())
        transforms_layout.addWidget(self.kernels_combo_box)

        h_layout = QHBoxLayout()

        self.radius_spin = QDoubleSpinBox()
        self.radius_spin.setRange(0.0, 1000.0)
        self.radius_spin.setValue(1.0)
        self.radius_spin.setPrefix("Radius: ")
        self.radius_spin.valueChanged.connect(self.update_kernel_info)
        h_layout.addWidget(self.radius_spin)

        self.physical_unit_check = QCheckBox("Physical units")
        self.physical_unit_check.stateChanged.connect(self.update_kernel_info)
        h_layout.addWidget(self.physical_unit_check)

        transforms_layout.addLayout(h_layout)

        self.kernel_info_label = QLabel("")
        transforms_layout.addWidget(self.kernel_info_label)

        transforms_layout.addSpacing(10)

        self.edt_button = QPushButton("Euclidean Distance Transform")
        transforms_layout.addWidget(self.edt_button)
        self.edt_button.clicked.connect(self.euclidean_distance_transform)

        transforms_layout.addSpacing(10)

        h_layout = QHBoxLayout()
        self.gray_opening_button = QPushButton("Gray Opening")
        h_layout.addWidget(self.gray_opening_button)
        self.gray_opening_button.clicked.connect(self.gray_opening)
        self.gray_closing_button = QPushButton("Gray Closing")
        h_layout.addWidget(self.gray_closing_button)
        self.gray_closing_button.clicked.connect(self.gray_closing)
        transforms_layout.addLayout(h_layout)

        h_layout = QHBoxLayout()
        self.white_tophat_button = QPushButton("White Tophat")
        h_layout.addWidget(self.white_tophat_button)
        self.white_tophat_button.clicked.connect(self.white_tophat)
        self.black_tophat_button = QPushButton("Black Tophat")
        h_layout.addWidget(self.black_tophat_button)
        self.black_tophat_button.clicked.connect(self.black_tophat)
        transforms_layout.addLayout(h_layout)

        h_layout = QHBoxLayout()
        self.closing_button = QPushButton("Binary Closing")
        h_layout.addWidget(self.closing_button)
        self.closing_button.clicked.connect(self.binary_closing)
        self.opening_button = QPushButton("Binary Opening")
        h_layout.addWidget(self.opening_button)
        self.opening_button.clicked.connect(self.binary_opening)
        transforms_layout.addLayout(h_layout)

        h_layout = QHBoxLayout()
        self.erosion_button = QPushButton("Binary Erosion")
        h_layout.addWidget(self.erosion_button)
        self.erosion_button.clicked.connect(self.binary_erosion)
        self.dilation_button = QPushButton("Binary Dilation")
        h_layout.addWidget(self.dilation_button)
        self.dilation_button.clicked.connect(self.binary_dilation)
        transforms_layout.addLayout(h_layout)

        self.update_kernel_info()

    def init_labeling_ui(self, layout):
        labeling_group = QGroupBox("Labeling")
        labeling_layout = QVBoxLayout()
        labeling_group.setLayout(labeling_layout)

        h_layout = QHBoxLayout()
        self.connectivity_spin = QDoubleSpinBox()
        self.connectivity_spin.setRange(1, 3)
        self.connectivity_spin.setValue(1)
        self.connectivity_spin.setPrefix("Connectivity: ")
        h_layout.addWidget(self.connectivity_spin)

        self.label_button = QPushButton("Islands labeling")
        self.label_button.clicked.connect(self.apply_labeling)
        h_layout.addWidget(self.label_button)
        labeling_layout.addLayout(h_layout)

        h_layout = QHBoxLayout()
        self.seeds_combobox = QComboBox()
        self.seeds_combobox.addItems(["---"])
        h_layout.addWidget(self.seeds_combobox)

        self.seeded_watershed_button = QPushButton("Seeded Watershed")
        self.seeded_watershed_button.clicked.connect(self.apply_seeded_watershed)
        h_layout.addWidget(self.seeded_watershed_button)

        labeling_layout.addLayout(h_layout)

        layout.addWidget(labeling_group)
    
    def on_method_update(self):
        if self.method_combobox.currentText() == "Manual":
            self.lower_spin.setEnabled(True)
            self.upper_spin.setEnabled(True)
        else:
            self.lower_spin.setEnabled(False)
            self.upper_spin.setEnabled(False)
        l, u = self.compute_threshold()
        if l is None or u is None:
            return
        self.lower_spin.setValue(l)
        self.upper_spin.setValue(u)
    
    def on_dark_bg_update(self):
        l, u = self.compute_threshold()
        if l is None or u is None:
            return
        self.lower_spin.setValue(l)
        self.upper_spin.setValue(u)

    def compute_threshold(self, f=-1):
        layer = self.viewer.layers.selection.active
        if layer is None:
            return (None, None)
        data = layer.data
        if self.individual_frames_check.isChecked() and data.ndim == 4:
            p = int(self.viewer.dims.point[0]) if f == -1 else f
            data = data[p]
        method_name = self.method_combobox.currentText()
        if method_name == "Manual":
            low = self.lower_spin.value()
            high = self.upper_spin.value()
            return (low, high)
        method = self.threshold_methods[method_name]
        thresh = method(data)
        return (thresh, data.max()) if self.dark_bg_check.isChecked() else (data.min(), thresh)

    def apply_threshold(self):
        layer = self.viewer.layers.selection.active
        if layer is None:
            return
        data = layer.data
        if data.ndim == 4 and self.individual_frames_check.isChecked():
            mask = np.zeros(data.shape, dtype=bool)
            for f in range(data.shape[0]):
                l, u = self.compute_threshold(f)
                mask[f] = (data[f] >= l) & (data[f] <= u)
        else:
            l = self.lower_spin.value()
            u = self.upper_spin.value()
            mask = (data >= l) & (data <= u)
        name = layer.name + f" {self.method_combobox.currentText()} mask"
        if name in self.viewer.layers:
            self.viewer.layers[name].data = mask.astype(np.uint8)
        else:
            self.viewer.add_labels(mask.astype(np.uint8), name=name, scale=layer.scale, units=layer.units)

    def apply_labeling(self):
        layer = self.viewer.layers.selection.active
        if layer is None:
            return
        data = layer.data
        connectivity = int(self.connectivity_spin.value())
        if data.ndim == 4:
            labeled = np.zeros(data.shape, dtype=np.uint16)
            for f in range(data.shape[0]):
                labeled[f] = label(data[f], connectivity=connectivity)
        else:
            labeled = label(data, connectivity=connectivity)
        name = layer.name + f" labeled"
        if name in self.viewer.layers:
            self.viewer.layers[name].data = labeled
        else:
            self.viewer.add_labels(labeled, name=name, scale=layer.scale, units=layer.units)

    def _get_layer_names(self):
        try:
            return [ly.name for ly in self.viewer.layers if hasattr(ly, 'symbol')]
        except Exception:
            return []
    
    def _set_combo_safely(self, combo: QComboBox, text: str):
        idx = combo.findText(text)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        else:
            combo.setCurrentIndex(0)

    def _populate_layer_combo(self, combo: QComboBox, neutral=NEUTRAL):
        current = combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        combo.addItem(neutral)
        for name in self._get_layer_names():
            combo.addItem(name)
        self._set_combo_safely(combo, current)
        combo.blockSignals(False)

    def refresh_layer_names(self):
        """Call this to refresh all comboboxes with current viewer layers."""
        self._populate_layer_combo(self.seeds_combobox, neutral=NEUTRAL)

    def compute_rz(self, radius_yx, anisotropy):
        if not (0.0 <= anisotropy <= 1.0):
            return 1.0
        if radius_yx < 0:
            return 1.0
        r_z = max(1, int(np.ceil(radius_yx * anisotropy)))
        return r_z
    
    def cube(self, radius_yx, anisotropy):
        r_yx = int(radius_yx)
        r_z = self.compute_rz(r_yx, anisotropy)

        sz = int(2 * r_z + 1 if r_z > 0 else 1)
        sy = int(2 * r_yx + 1)
        sx = int(2 * r_yx + 1)

        return np.ones((sz, sy, sx), dtype=bool)

    def ball(self, radius_yx, anisotropy):
        r_yx = int(radius_yx)
        r_z = self.compute_rz(r_yx, anisotropy)
        zz, yy, xx = np.ogrid[-r_z:r_z+1, -r_yx:r_yx+1, -r_yx:r_yx+1]
        dist = (zz / float(r_z))**2 + (yy / float(r_yx))**2 + (xx / float(r_yx))**2
        footprint = dist <= 1.0
        return footprint
    
    def update_kernel_info(self):
        layer = self.viewer.layers.selection.active
        if layer is None:
            self.kernel_info_label.setText("Z: --, Y: --, X: --")
            return
        r_yx = self.radius_spin.value()
        anisotropy = layer.scale[-1] / layer.scale[-3]
        if self.physical_unit_check.isChecked():
            r_yx = self.radius_spin.value() / layer.scale[-1]
        r_z = self.compute_rz(r_yx, anisotropy)
        self.kernel_info_label.setText(f"Z: {r_z}, Y: {int(r_yx)}, X: {int(r_yx)}")
    
    def get_kernel(self):
        layer = self.viewer.layers.selection.active
        if layer is None:
            return None
        yx = self.radius_spin.value()
        anisotropy = layer.scale[-1] / layer.scale[-3]
        if self.physical_unit_check.isChecked():
            yx = self.radius_spin.value() / layer.scale[-1]
        kernel_func = self.kernels[self.kernels_combo_box.currentText()]
        kernel = kernel_func(yx, anisotropy)
        return kernel
        
    def apply_seeded_watershed(self):
        pass

    def euclidean_distance_transform(self):
        layer = self.viewer.layers.selection.active
        if layer is None:
            return
        data = layer.data
        scale = layer.scale[-3:]
        if data.ndim == 4:
            edt_data = np.zeros(data.shape, dtype=np.float32)
            for f in range(data.shape[0]):
                edt_data[f] = distance_transform_edt(data[f], sampling=scale)
        else:
            edt_data = distance_transform_edt(data, sampling=scale)
        name = layer.name + " EDT"
        if name in self.viewer.layers:
            self.viewer.layers[name].data = edt_data
        else:
            self.viewer.add_image(edt_data, name=name, scale=layer.scale, units=layer.units)

    def gray_opening(self):
        layer = self.viewer.layers.selection.active
        if layer is None:
            return
        data = layer.data
        kernel = self.get_kernel()
        if kernel is None:
            return
        if data.ndim == 4:
            opened_data = np.zeros(data.shape, dtype=data.dtype)
            for f in range(data.shape[0]):
                opened_data[f] = grey_opening(data[f], footprint=kernel)
        else:
            opened_data = grey_opening(data, footprint=kernel)
        name = layer.name + " gray opening"
        if name in self.viewer.layers:
            self.viewer.layers[name].data = opened_data
        else:
            self.viewer.add_image(opened_data, name=name, scale=layer.scale, units=layer.units)

    def gray_closing(self):
        layer = self.viewer.layers.selection.active
        if layer is None:
            return
        data = layer.data
        kernel = self.get_kernel()
        if kernel is None:
            return
        if data.ndim == 4:
            closed_data = np.zeros(data.shape, dtype=data.dtype)
            for f in range(data.shape[0]):
                closed_data[f] = grey_closing(data[f], footprint=kernel)
        else:
            closed_data = grey_closing(data, footprint=kernel)
        name = layer.name + " gray closing"
        if name in self.viewer.layers:
            self.viewer.layers[name].data = closed_data
        else:
            self.viewer.add_image(closed_data, name=name, scale=layer.scale, units=layer.units)

    def white_tophat(self):
        layer = self.viewer.layers.selection.active
        if layer is None:
            return
        data = layer.data
        kernel = self.get_kernel()
        if kernel is None:
            return
        if data.ndim == 4:
            tophat_data = np.zeros(data.shape, dtype=data.dtype)
            for f in range(data.shape[0]):
                tophat_data[f] = white_tophat(data[f], footprint=kernel)
        else:
            tophat_data = white_tophat(data, footprint=kernel)
        name = layer.name + " white tophat"
        if name in self.viewer.layers:
            self.viewer.layers[name].data = tophat_data
        else:
            self.viewer.add_image(tophat_data, name=name, scale=layer.scale, units=layer.units)

    def black_tophat(self):
        layer = self.viewer.layers.selection.active
        if layer is None:
            return
        data = layer.data
        kernel = self.get_kernel()
        if kernel is None:
            return
        if data.ndim == 4:
            tophat_data = np.zeros(data.shape, dtype=data.dtype)
            for f in range(data.shape[0]):
                tophat_data[f] = black_tophat(data[f], footprint=kernel)
        else:
            tophat_data = black_tophat(data, footprint=kernel)
        name = layer.name + " black tophat"
        if name in self.viewer.layers:
            self.viewer.layers[name].data = tophat_data
        else:
            self.viewer.add_image(tophat_data, name=name, scale=layer.scale, units=layer.units)

    def binary_closing(self):
        layer = self.viewer.layers.selection.active
        if layer is None:
            return
        data = layer.data
        kernel = self.get_kernel()
        if kernel is None:
            return
        if data.ndim == 4:
            closed_data = np.zeros(data.shape, dtype=data.dtype)
            for f in range(data.shape[0]):
                closed_data[f] = grey_closing(data[f], footprint=kernel).astype(data.dtype)
        else:
            closed_data = grey_closing(data, footprint=kernel).astype(data.dtype)
        name = layer.name + " binary closing"
        if name in self.viewer.layers:
            self.viewer.layers[name].data = closed_data
        else:
            self.viewer.add_labels(closed_data, name=name, scale=layer.scale, units=layer.units)

    def binary_opening(self):
        layer = self.viewer.layers.selection.active
        if layer is None:
            return
        data = layer.data
        kernel = self.get_kernel()
        if kernel is None:
            return
        if data.ndim == 4:
            opened_data = np.zeros(data.shape, dtype=data.dtype)
            for f in range(data.shape[0]):
                opened_data[f] = grey_opening(data[f], footprint=kernel).astype(data.dtype)
        else:
            opened_data = grey_opening(data, footprint=kernel).astype(data.dtype)
        name = layer.name + " binary opening"
        if name in self.viewer.layers:
            self.viewer.layers[name].data = opened_data
        else:
            self.viewer.add_labels(opened_data, name=name, scale=layer.scale, units=layer.units)

    def binary_erosion(self):
        layer = self.viewer.layers.selection.active
        if layer is None:
            return
        data = layer.data
        kernel = self.get_kernel()
        if kernel is None:
            return
        if data.ndim == 4:
            eroded_data = np.zeros(data.shape, dtype=data.dtype)
            for f in range(data.shape[0]):
                eroded_data[f] = grey_opening(data[f], footprint=kernel).astype(data.dtype)
        else:
            eroded_data = grey_opening(data, footprint=kernel).astype(data.dtype)
        name = layer.name + " binary erosion"
        if name in self.viewer.layers:
            self.viewer.layers[name].data = eroded_data
        else:
            self.viewer.add_labels(eroded_data, name=name, scale=layer.scale, units=layer.units)

    def binary_dilation(self):
        layer = self.viewer.layers.selection.active
        if layer is None:
            return
        data = layer.data
        kernel = self.get_kernel()
        if kernel is None:
            return
        if data.ndim == 4:
            dilated_data = np.zeros(data.shape, dtype=data.dtype)
            for f in range(data.shape[0]):
                dilated_data[f] = grey_closing(data[f], footprint=kernel).astype(data.dtype)
        else:
            dilated_data = grey_closing(data, footprint=kernel).astype(data.dtype)
        name = layer.name + " binary dilation"
        if name in self.viewer.layers:
            self.viewer.layers[name].data = dilated_data
        else:
            self.viewer.add_labels(dilated_data, name=name, scale=layer.scale, units=layer.units)


def loose_launch():
    viewer = napari.Viewer()
    widget = MaskUtils(viewer=viewer)
    viewer.window.add_dock_widget(widget)

    napari.run()

if __name__ == "__main__":
    loose_launch()