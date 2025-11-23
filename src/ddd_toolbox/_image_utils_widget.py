import os
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QTabWidget, QInputDialog,
                            QGroupBox, QHBoxLayout, QLabel, QToolButton, QButtonGroup,
                            QComboBox, QCheckBox, QLineEdit, QSpinBox,
                            QPushButton, QFileDialog, QDoubleSpinBox
)
from qtpy.QtCore import Qt, QThread

from scipy.ndimage import zoom

import napari
from napari.utils import progress

import numpy as np

button_style = """
QToolButton {
    border: 1px solid #666;
    border-radius: 4px;
    padding: 4px 8px;
    background-color: transparent;
    color: white;
}
QToolButton:hover {
    background-color: #444;
}
QToolButton:checked {
    background-color: #007acc;   /* Active button color */
    border: 1px solid #005f99;
}
"""

NEUTRAL = "---------"

class ImageUtilsWidget(QWidget):
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.layer_pools = []
        self.init_ui()
        self.viewer.layers.events.inserted.connect(lambda e: self.refresh_layer_names())
        self.viewer.layers.events.removed.connect(lambda e: self.refresh_layer_names())
        self.viewer.layers.events.reordered.connect(lambda e: self.refresh_layer_names())
    
    def init_ui(self):
        layout = QVBoxLayout()
        gap = 10
        self.setLayout(layout)

        # SPLIT AXIS
        h_layout = QHBoxLayout()
        self.axis_spin = QSpinBox()
        self.axis_spin.setMinimum(0)
        self.axis_spin.setValue(0)
        h_layout.addWidget(self.axis_spin)
        self.split_button = QPushButton("Split axis")
        self.split_button.clicked.connect(self.split_axis)
        h_layout.addWidget(self.split_button)
        layout.addLayout(h_layout)

        layout.addSpacing(gap)

        # NORMALIZE
        self.normalize_button = QPushButton("Normalize values")
        self.normalize_button.clicked.connect(self.normalize_values)
        layout.addWidget(self.normalize_button)

        layout.addSpacing(gap)

        # RESLICE
        labels = ["-Z", "+Z", "-Y", "+Y", "-X", "+X"]
        self.group_reslice = QButtonGroup(self)
        self.group_reslice.setExclusive(True)
        
        reslice_layout = QHBoxLayout()
        self.buttons_reslice = []
        for text in labels:
            btn = QToolButton()
            btn.setText(text)
            btn.setCheckable(True)
            btn.setStyleSheet(button_style)
            self.buttons_reslice.append(btn)
            self.group_reslice.addButton(btn)
            reslice_layout.addWidget(btn)
        self.reslice_button = QPushButton("Reslice")
        self.reslice_button.clicked.connect(self.reslice)
        reslice_layout.addWidget(self.reslice_button)
        layout.addLayout(reslice_layout)

        layout.addSpacing(gap)

        # TYPE CAST
        labels = ["uint8", "uint16", "uint32", "float"]
        cast_layout = QHBoxLayout()
        layout.addLayout(cast_layout)
        self.group_typecast = QButtonGroup(self)
        self.group_typecast.setExclusive(True)
        for text in labels:
            btn = QToolButton()
            btn.setText(text)
            btn.setCheckable(True)
            btn.setStyleSheet(button_style)
            btn.clicked.connect(lambda checked, t=text: print(t))
            self.group_typecast.addButton(btn)
            cast_layout.addWidget(btn)
        self.typecast_button = QPushButton("Type Cast")
        self.typecast_button.clicked.connect(self.type_cast)
        cast_layout.addWidget(self.typecast_button)
        layout.addSpacing(gap)

        # RESAMPLE ISOTROPIC
        h_layout = QHBoxLayout()
        self.interpolate_resample_checkbox = QCheckBox("Interpolate")
        self.interpolate_resample_checkbox.setChecked(True)
        h_layout.addWidget(self.interpolate_resample_checkbox)
        self.resample_button = QPushButton("Resample isotropic")
        self.resample_button.clicked.connect(self.apply_resample_isotropic)
        h_layout.addWidget(self.resample_button)
        layout.addLayout(h_layout)

        h_layout = QHBoxLayout()
        self.translate_checkbox = QCheckBox("Translate")
        h_layout.addWidget(self.translate_checkbox)
        self.crop_selection_combobox = QComboBox()
        self.layer_pools.append((self.crop_selection_combobox, "shape_type"))
        h_layout.addWidget(self.crop_selection_combobox)
        self.crop_to_selection_button = QPushButton("Crop")
        h_layout.addWidget(self.crop_to_selection_button)
        layout.addLayout(h_layout)
        self.crop_to_selection_button.clicked.connect(self.crop_to_selection)

    def split_axis(self):
        a = int(self.axis_spin.value())
        layer = self.viewer.layers.selection.active
        if layer is None:
            return
        data = layer.data
        if data.ndim <= a:
            return
        size = data.shape[a]
        components = np.split(data, size, axis=a)
        scale = [l for i, l in enumerate(layer.scale) if i != a]
        units = [l for i, l in enumerate(layer.units) if i != a]
        for i, c in enumerate(components):
            self.viewer.add_image(np.squeeze(c), name=f"{layer.name} #{i}", scale=scale, units=units, blending='additive')
        self.viewer.layers.remove(layer)

    def normalize_values(self):
        layer = self.viewer.layers.selection.active
        if layer is None:
            return
        
        target = None
        if layer.data.dtype == np.uint8:
            target = (0, 255)
        elif layer.data.dtype == np.int8:
            target = (-128, 127)
        elif layer.data.dtype == np.uint16:
            target = (0, 65535)
        elif layer.data.dtype == np.int16:
            target = (-32768, 32767)
        elif layer.data.dtype == np.int32:
            target = (-2147483648, 2147483647)
        elif layer.data.dtype == np.float32 or layer.data.dtype == np.float64:
            target = (0.0, 1.0)

        if target is None:
            return
        
        data = layer.data.astype(np.float32)
        dmin = data.min()
        dmax = data.max()
        if dmax - dmin == 0:
            return
        norm_data = (data - dmin) / (dmax - dmin)
        norm_data = norm_data * (target[1] - target[0]) + target[0]
        layer.data = norm_data.astype(layer.data.dtype)

    def type_cast(self):
        types = {
            "uint8": np.uint8,
            "uint16": np.uint16,
            "uint32": np.uint32,
            "float": np.float32
        }
        types_max = {
            "uint8": 255,
            "uint16": 65535,
            "uint32": 4294967295,
            "float": 1.0
        }
        selected_button = self.group_typecast.checkedButton()
        if selected_button is None:
            return
        target_type = types[selected_button.text()]
        target_max = types_max[selected_button.text()]
        layer = self.viewer.layers.selection.active
        if layer is None:
            return
        data = layer.data.astype(np.float32)
        dmin = data.min()
        dmax = data.max()
        if dmax - dmin == 0:
            return
        norm_data = (data - dmin) / (dmax - dmin)
        norm_data = norm_data * target_max
        layer.data = norm_data.astype(target_type)
        layer.contrast_limits = (0, target_max)

    def view_along_axis(self, img, axis: str):
        if img.ndim != 3:
            raise ValueError("Input image must be 3D (Z, Y, X).")

        axis = axis.upper().replace(" ", "")
        if axis not in ["+Z", "-Z", "+Y", "-Y", "+X", "-X"]:
            raise ValueError("Axis must be one of: +Z, -Z, +Y, -Y, +X, -X")

        if axis == "+Z":
            return img

        if axis == "-Z":
            return img[::-1, :, :]

        if axis == "+Y":
            out = np.swapaxes(img, 0, 1) # (Y, Z, X)
            return out

        if axis == "-Y":
            out = np.swapaxes(img, 0, 1) # (Y, Z, X)
            return out[::-1, :, :] # flip along first axis

        if axis == "+X":
            out = np.swapaxes(img, 0, 2) # (X, Y, Z)
            return out

        if axis == "-X":
            out = np.swapaxes(img, 0, 2) # (X, Y, Z)
            return out[::-1, :, :]

        return img

    def reslice(self):
        selected_button = self.group_reslice.checkedButton()
        if selected_button is None:
            return
        direction = selected_button.text()
        layer = self.viewer.layers.selection.active
        if layer is None:
            return
        all_data = layer.data if layer.ndim == 4 else layer.data[np.newaxis, ...]
        transformed = []
        z_ax, y_ax, x_ax = -3, -2, -1
        new_scale = list(layer.scale)

        # Determine new scale and units based on direction
        new_units = list(layer.units)
        new_scale = list(layer.scale)
        if direction in ["+Z", "-Z"]:
            pass
        elif direction in ["+Y", "-Y"]:
            new_scale[z_ax] = layer.scale[y_ax]
            new_units[z_ax] = layer.units[y_ax]
            # New Y scale is old Z scale
            new_scale[y_ax] = layer.scale[z_ax]
            new_units[y_ax] = layer.units[z_ax]
        elif direction in ["+X", "-X"]:
            # New Z scale is old X scale
            new_scale[z_ax] = layer.scale[x_ax]
            new_units[z_ax] = layer.units[x_ax]
            # New X scale is old Z scale
            new_scale[x_ax] = layer.scale[z_ax]
            new_units[x_ax] = layer.units[z_ax]

        for frame_idx in range(all_data.shape[0]):
            v = all_data[frame_idx]
            t = self.view_along_axis(v, direction)
            transformed.append(t)

        result = np.stack(transformed, axis=0)
        name = f"{layer.name} resliced {direction}"

        if name in self.viewer.layers:
            self.viewer.layers[name].data = result if layer.ndim == 4 else result[0]
        else:
            self.viewer.add_image(
                result if layer.ndim == 4 else result[0],
                name=name,
                scale=new_scale,
                units=new_units,
                blending='additive',
                metadata=layer.metadata.copy()
            )

    def isotropic_resample(self, img, spacing):
        if img.ndim != 3:
            raise ValueError("Input image must be 3D (Z, Y, X).")
        sz, sy, sx = spacing
        spacing = np.asarray([sz, sy, sx], dtype=float)
        iso = spacing.min()
        scale = spacing / iso
        new_shape = np.round(np.array(img.shape) * scale).astype(int)
        zoom_factors = new_shape / np.array(img.shape, dtype=float)
        out = zoom(img, zoom_factors, order=1 if self.interpolate_resample_checkbox.isChecked() else 0)
        return out, iso

    def apply_resample_isotropic(self):
        layer = self.viewer.layers.selection.active
        if layer is None:
            return
        data = layer.data if layer.ndim == 4 else layer.data[np.newaxis, ...]
        scale = layer.scale[-3:]
        transformed = []

        iso = 1
        for frame_idx in range(data.shape[0]):
            v = data[frame_idx]
            t, iso = self.isotropic_resample(v, scale)
            transformed.append(t)

        result = np.stack(transformed, axis=0)
        new_scale = list(layer.scale)
        new_scale[-3:] = [iso, iso, iso]
        name = f"{layer.name} resampled isotropic"

        if name in self.viewer.layers:
            self.viewer.layers[name].data = result if layer.ndim == 4 else result[0]
        else:
            self.viewer.add_image(
                result if layer.ndim == 4 else result[0],
                name=name,
                scale=new_scale,
                units=layer.units,
                blending='additive',
                metadata=layer.metadata.copy()
            )

    def get_range_from_text(self, default, text, lower, upper):
        values = text.split('-')
        if len(values) != 2:
            return default
        try:
            start = int(values[0].strip())
            end = int(values[1].strip())
        except Exception:
            return default
        if start < lower or end > upper or start > end:
            return default
        return (start, end)
    
    def crop_to_selection(self):
        layer = self.viewer.layers.selection.active
        if layer is None or not hasattr(layer, "colormap"):
            return
        data = layer.data if layer.ndim == 4 else layer.data[np.newaxis, ...]
        shape_name = self.crop_selection_combobox.currentText()
        if shape_name not in self.viewer.layers:
            return
        shape_layer = self.viewer.layers[shape_name]
        if shape_layer.ndim != layer.ndim:
            return
        true_len = min(len(shape_layer.data), len(shape_layer.shape_type))
        stype = shape_layer.shape_type[true_len-1]
        if stype != "rectangle":
            return
        target_range = (1, data.shape[0])
        if layer.ndim == 4:
            txt_labels = QInputDialog.getText(self,  'Range of frames', 'Enter the interval of frames to crop (start-end):', text=f"1-{data.shape[0]}")
            if txt_labels[1]:
                target_range = self.get_range_from_text((1, data.shape[0]), txt_labels[0], 1, data.shape[0])
        rect = shape_layer.data[true_len-1].T
        min_y = min(rect[-2])
        max_y = max(rect[-2])
        min_x = min(rect[-1])
        max_x = max(rect[-1])
        if min_x < 0:
            min_x = 0
        if min_y < 0:
            min_y = 0
        if max_x > layer.data.shape[-1]:
            max_x = layer.data.shape[-1]
        if max_y > layer.data.shape[-2]:
            max_y = layer.data.shape[-2]
        transformed = []
        for frame_idx in range(data.shape[0]):
            v = data[frame_idx]
            t = v[..., int(min_y):int(max_y), int(min_x):int(max_x)]
            transformed.append(t)
        transformed = transformed[target_range[0]-1:target_range[1]]
        result = np.stack(transformed, axis=0)
        name = f"{layer.name} cropped"
        if name in self.viewer.layers:
            l = self.viewer.layers[name]
            l.data = result if layer.ndim == 4 else result[0]
        else:
            l = self.viewer.add_image(
                result if layer.ndim == 4 else result[0],
                name=name,
                scale=layer.scale,
                units=layer.units,
                blending='additive',
                metadata=layer.metadata.copy()
            )
        if self.translate_checkbox.isChecked():
            translation = [0 for _ in range(l.ndim)]
            translation[-2] = min_y
            translation[-1] = min_x
            l.translate = translation

    def _get_layer_names(self, ppt):
        try:
            return [ly.name for ly in self.viewer.layers if hasattr(ly, ppt)]
        except Exception:
            return []

    def _set_combo_safely(self, combo: QComboBox, text: str):
        idx = combo.findText(text)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        else:
            # fall back to neutral
            combo.setCurrentIndex(0)

    def _populate_layer_combo(self, pair: QComboBox, neutral=NEUTRAL):
        combo, ppt = pair
        current = combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        combo.addItem(neutral)
        for name in self._get_layer_names(ppt):
            combo.addItem(name)
        self._set_combo_safely(combo, current)
        combo.blockSignals(False)

    def refresh_layer_names(self):
        """Call this to refresh all comboboxes with current viewer layers."""
        for combo in self.layer_pools:
            self._populate_layer_combo(combo, neutral=NEUTRAL)


def loose_launch():
    viewer = napari.Viewer()
    widget = ImageUtilsWidget(viewer=viewer)
    viewer.window.add_dock_widget(widget)

    napari.run()

if __name__ == "__main__":
    loose_launch()