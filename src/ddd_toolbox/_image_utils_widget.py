import os
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QTabWidget,
                            QGroupBox, QHBoxLayout, QLabel, QToolButton, QButtonGroup,
                            QComboBox, QCheckBox, QLineEdit, QSpinBox,
                            QPushButton, QFileDialog, QDoubleSpinBox
)
from qtpy.QtCore import Qt, QThread

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

def perm_with_spatial(nd, axes):
    perm = list(range(nd))
    base_positions = sorted(axes)
    for pos, ax in zip(base_positions, axes):
        perm[pos] = ax
    return perm

class ImageUtilsWidget(QWidget):
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.init_ui()
    
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
        self.resample_button = QPushButton("Resample isotropic")
        layout.addWidget(self.resample_button)

        layout.addSpacing(gap)

        # HARD TRANSFORMATIONS
        self.group_transforms = QGroupBox("Transforms")
        v_layout = QVBoxLayout()
        self.group_transforms.setLayout(v_layout)

        h_layout = QHBoxLayout()
        self.scale_z = QDoubleSpinBox()
        self.scale_z.setPrefix("Z:")
        self.scale_z.setValue(1.0)
        h_layout.addWidget(self.scale_z)

        self.scale_y = QDoubleSpinBox()
        self.scale_y.setPrefix("Y:")
        self.scale_y.setValue(1.0)
        h_layout.addWidget(self.scale_y)

        self.scale_x = QDoubleSpinBox()
        self.scale_x.setPrefix("X:")
        self.scale_x.setValue(1.0)
        h_layout.addWidget(self.scale_x)

        self.transform_button = QPushButton("Scale")
        h_layout.addWidget(self.transform_button)
        v_layout.addLayout(h_layout)

        h_layout = QHBoxLayout()
        self.rotate_z = QDoubleSpinBox()
        self.rotate_z.setPrefix("Z:")
        self.rotate_z.setValue(0.0)
        h_layout.addWidget(self.rotate_z)

        self.rotate_y = QDoubleSpinBox()
        self.rotate_y.setPrefix("Y:")
        self.rotate_y.setValue(0.0)
        h_layout.addWidget(self.rotate_y)

        self.rotate_x = QDoubleSpinBox()
        self.rotate_x.setPrefix("X:")
        self.rotate_x.setValue(0.0)
        h_layout.addWidget(self.rotate_x)

        self.rotate_button = QPushButton("Rotate")
        h_layout.addWidget(self.rotate_button)
        v_layout.addLayout(h_layout)

        layout.addWidget(self.group_transforms)

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

    def reslice(self):
        selected_button = self.group_reslice.checkedButton()
        if selected_button is None:
            return
        direction = selected_button.text()
        layer = self.viewer.layers.selection.active
        if layer is None:
            return
        v = layer.data
        nd = v.ndim
        z_ax, y_ax, x_ax = nd - 3, nd - 2, nd - 1

        if direction == '+Z':
            layer.data = v  # No change

        elif direction == '-Z':
            layer.data = np.flip(v, axis=z_ax)

        elif direction == '+Y':
            perm = perm_with_spatial(nd, [y_ax, z_ax, x_ax])
            layer.data = np.transpose(v, axes=perm)

        elif direction == '-Y':
            v2 = np.flip(v, axis=y_ax)
            perm = perm_with_spatial(nd, [y_ax, z_ax, x_ax])
            layer.data = np.transpose(v2, axes=perm)
            layer.scale = tuple(layer.scale[i] for i in range(nd) if i != y_ax) + (layer.scale[y_ax],)
            layer.units = tuple(layer.units[i] for i in range(nd) if i != y_ax) + (layer.units[y_ax],)

        elif direction == '+X':
            perm = perm_with_spatial(nd, [x_ax, y_ax, z_ax])
            layer.data = np.transpose(v, axes=perm)
            layer.scale = tuple(layer.scale[i] for i in range(nd) if i != x_ax) + (layer.scale[x_ax],)
            layer.units = tuple(layer.units[i] for i in range(nd) if i != x_ax) + (layer.units[x_ax],)

        elif direction == '-X':
            v2 = np.flip(v, axis=x_ax)
            perm = perm_with_spatial(nd, [x_ax, y_ax, z_ax])
            layer.data = np.transpose(v2, axes=perm)
            layer.scale = tuple(layer.scale[i] for i in range(nd) if i != x_ax) + (layer.scale[x_ax],)
            layer.units = tuple(layer.units[i] for i in range(nd) if i != x_ax) + (layer.units[x_ax],)


def loose_launch():
    viewer = napari.Viewer()
    widget = ImageUtilsWidget(viewer=viewer)
    viewer.window.add_dock_widget(widget)

    napari.run()

if __name__ == "__main__":
    loose_launch()