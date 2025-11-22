from qtpy.QtWidgets import (QWidget, QVBoxLayout, QTabWidget,
                            QGroupBox, QHBoxLayout, QLabel, 
                            QComboBox, QCheckBox, QLineEdit, 
                            QPushButton, QFileDialog, QDoubleSpinBox,
                            QInputDialog, QDialog, QFormLayout)
from qtpy.QtCore import Qt, QThread

import napari
from napari.utils import progress
from napari.utils import Colormap

import numpy as np

from skimage.segmentation import clear_border
from skimage.measure import regionprops
from ddd_toolbox.results_table import LabelsPropertiesResultsTable

NEUTRAL = "--------"

class LabelsOperations(QWidget):
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.layer_pools = []
        self.init_ui()

        self.results_tables = {}

        self.viewer.layers.events.inserted.connect(lambda e: self.refresh_layer_names())
        self.viewer.layers.events.removed.connect(lambda e: self.refresh_layer_names())
        self.viewer.layers.events.reordered.connect(lambda e: self.refresh_layer_names())
    
    def init_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.label_operations_group = QGroupBox("Labels operations")
        layout.addWidget(self.label_operations_group)
        layout = QVBoxLayout()
        self.label_operations_group.setLayout(layout)

        h_layout = QHBoxLayout()

        self.select_labels_points_combobox = QComboBox()
        self.select_labels_points_combobox.addItem(NEUTRAL)
        self.layer_pools.append((self.select_labels_points_combobox, "symbol"))
        h_layout.addWidget(self.select_labels_points_combobox)

        self.select_labels_button = QPushButton("Select labels")
        self.select_labels_button.clicked.connect(self.select_labels)
        h_layout.addWidget(self.select_labels_button)
        
        layout.addLayout(h_layout)

        h_layout = QHBoxLayout()

        self.merge_labels_combobox = QComboBox()
        self.merge_labels_combobox.addItem(NEUTRAL)
        self.layer_pools.append((self.merge_labels_combobox, "shape_type"))
        h_layout.addWidget(self.merge_labels_combobox)

        self.merge_labels_button = QPushButton("Merge labels")
        self.merge_labels_button.clicked.connect(self.merge_labels)
        h_layout.addWidget(self.merge_labels_button)

        layout.addLayout(h_layout)

        self.keep_largest_button = QPushButton("Keep largest")
        self.keep_largest_button.clicked.connect(self.keep_largest)
        layout.addWidget(self.keep_largest_button)

        self.kill_border_labels_button = QPushButton("Kill borders")
        self.kill_border_labels_button.clicked.connect(self.kill_border_labels)
        layout.addWidget(self.kill_border_labels_button)

        h_layout = QHBoxLayout()

        self.remap_individual_check = QCheckBox("Individual frames")
        h_layout.addWidget(self.remap_individual_check)

        self.remap_labels_button = QPushButton("Remap labels")
        h_layout.addWidget(self.remap_labels_button)
        layout.addLayout(h_layout)

        layout.addSpacing(10)

        self.use_physical_units_check = QCheckBox("Use physical units")
        layout.addWidget(self.use_physical_units_check)

        h_layout = QHBoxLayout()

        self.intensity_channel_combobox = QComboBox()
        self.intensity_channel_combobox.addItem(NEUTRAL)
        self.layer_pools.append((self.intensity_channel_combobox, "colormap"))
        h_layout.addWidget(self.intensity_channel_combobox)

        self.measure_labels_button = QPushButton("Measure labels")
        self.measure_labels_button.clicked.connect(self.measure_labels)
        h_layout.addWidget(self.measure_labels_button)

        layout.addLayout(h_layout)

        self.assign_measure_button = QPushButton("Assign measurement to labels")
        layout.addWidget(self.assign_measure_button)

        self.labels_property_filter_button = QPushButton("Labels properties filtering")
        self.labels_property_filter_button.clicked.connect(self.labels_property_filter)
        layout.addWidget(self.labels_property_filter_button)
    
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
    
    def ask_settings(self, parent=None):
        dlg = QDialog(parent)
        dlg.setWindowTitle("Settings")

        form = QFormLayout()

        radius = QLineEdit("5")
        name = QLineEdit("Sample")
        enable = QCheckBox("Enable feature")

        form.addRow("Radius:", radius)
        form.addRow("Name:", name)
        form.addRow(enable)

        btn_ok = QPushButton("OK")
        btn_ok.clicked.connect(dlg.accept)

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(btn_ok)
        dlg.setLayout(layout)

        if dlg.exec_():
            return {
                "radius": int(radius.text()),
                "name": name.text(),
                "enabled": enable.isChecked(),
            }
        return None

    def labels_property_filter(self):
        settings = self.ask_settings(parent=self)
        if settings is None:
            return
        print("Settings:", settings)

    def keep_largest(self):
        layer = self.viewer.layers.selection.active
        if layer is None or not hasattr(layer, "selected_label"):
            return
        data = layer.data
        if data.ndim == 4:
            filtered = np.zeros_like(data)
            for t in range(data.shape[0]):
                labels, counts = np.unique(data[t], return_counts=True)
                if labels.size <= 1:
                    continue
                if int(labels[0]) == 0:
                    counts[0] = 0
                largest_label = labels[np.argmax(counts)]
                mask = (data[t] == largest_label).astype(data.dtype)
                filtered[t] = data[t] * mask
        else:
            labels, counts = np.unique(data, return_counts=True)
            if labels.size <= 1:
                return
            if int(labels[0]) == 0:
                counts[0] = 0
            largest_label = labels[np.argmax(counts)]
            mask = (data == largest_label).astype(data.dtype)
            filtered = data * mask
        name = layer.name + " largest"
        if name in self.viewer.layers:
            self.viewer.layers[name].data = filtered
        else:
            self.viewer.add_labels(filtered, name=name, scale=layer.scale, units=layer.units, metadata=layer.metadata.copy())

    def list_to_integers(self, text):
        try:
            parts = text.split(',')
            integers = set([int(p.strip()) for p in parts])
            return integers
        except Exception:
            return None

    def _select_labels(self, data, frame_labels):
        filtered = [np.zeros_like(data)] if data.ndim == 3 else np.zeros_like(data)
        input_data = [data] if data.ndim == 3 else data
        for frame_idx, labels in enumerate(frame_labels):
            mask = np.isin(input_data[frame_idx], list(labels)).astype(input_data[frame_idx].dtype)
            filtered[frame_idx] = input_data[frame_idx] * mask
        return filtered if data.ndim == 4 else np.concatenate(filtered, axis=0)

    def select_labels(self):
        layer = self.viewer.layers.selection.active
        if layer is None or not hasattr(layer, "selected_label"):
            return
        points_name = self.select_labels_points_combobox.currentText()
        frame_labels = []
        # If not point layer is selected, we ask for labels
        if points_name == NEUTRAL:
            txt_labels = QInputDialog.getText(self, 'Select labels', 'Enter labels to select (comma separated):')
            labels = self.list_to_integers(txt_labels[0])
            if labels is None:
                return
            if layer.ndim == 4:
                frame_labels = [labels for _ in range(layer.data.shape[0])]
            else:
                frame_labels = [labels]
        else:
            points_layer = self.viewer.layers[points_name]
            if points_layer is None or points_layer.ndim != layer.ndim:
                return
            if points_layer.ndim == 4: # if we have time, we take for each frame
                frame_labels = [set() for _ in range(layer.data.shape[0])]
                for co in points_layer.data:
                    frame = int(round(co[0]))
                    coord_int = tuple(int(round(x)) for x in co)
                    lbl = layer.data[coord_int]
                    if lbl != 0:
                        frame_labels[frame].add(lbl)
            else: # simple 3D points
                coords = points_layer.data
                labels = set()
                for c in coords:
                    coord_int = tuple(int(round(x)) for x in c)
                    lbl = layer.data[coord_int]
                    if lbl != 0:
                        labels.add(lbl)
                frame_labels.append(labels)
        filtered = self._select_labels(layer.data, frame_labels)
        name = layer.name + " selected labels"
        if name in self.viewer.layers:
            self.viewer.layers[name].data = filtered
        else:
            self.viewer.add_labels(filtered, name=name, scale=layer.scale, units=layer.units, metadata=layer.metadata.copy())

    def _merge_labels(self, data, frame_labels):
        merged = [np.zeros_like(data)] if data.ndim == 3 else np.zeros_like(data)
        input_data = [data] if data.ndim == 3 else data
        for frame_idx, labels_tuples in enumerate(frame_labels):
            label_map = {}
            for lbls in labels_tuples:
                new_label = np.min(lbls)
                for lbl in lbls:
                    label_map[lbl] = new_label
            output_frame = np.zeros_like(input_data[frame_idx])
            for lbl_in, lbl_out in label_map.items():
                output_frame[input_data[frame_idx] == lbl_in] = lbl_out
            # Copy unchanged labels
            unique_labels = np.unique(input_data[frame_idx])
            for ul in unique_labels:
                if ul != 0 and ul not in label_map:
                    output_frame[input_data[frame_idx] == ul] = ul
            merged[frame_idx] = output_frame
        return merged if data.ndim == 4 else np.concatenate(merged, axis=0)

    def merge_labels(self):
        layer = self.viewer.layers.selection.active
        if layer is None or not hasattr(layer, "selected_label"):
            return
        shape_type_name = self.merge_labels_combobox.currentText()
        frame_labels = []
        # If not shape_type layer is selected, we ask for labels
        if shape_type_name == NEUTRAL:
            txt_labels = QInputDialog.getText(self, 'Merge labels', 'Enter labels to merge (comma separated):')
            labels = self.list_to_integers(txt_labels[0])
            if labels is None or len(labels) < 2:
                return
            labels = tuple(labels)
            if layer.ndim == 4:
                frame_labels = [set([labels]) for _ in range(layer.data.shape[0])]
            else:
                frame_labels = [set([labels])]
        else:
            shape_type_layer = self.viewer.layers[shape_type_name]
            if shape_type_layer is None or shape_type_layer.ndim != layer.ndim:
                return
            allowed = set(['path', 'line'])
            found = set(shape_type_layer.shape_type)
            if len(found.intersection(allowed)) != len(found):
                return
            if shape_type_layer.ndim == 4: # if we have time, we take for each frame
                frame_labels = [set() for _ in range(layer.data.shape[0])] # each set contains tuples of labels to merge
                for polyline in shape_type_layer.data:
                    t = int(round(polyline[0][0]))
                    lbls = []
                    for point in polyline:
                        coord_int = tuple(int(round(x)) for x in point)
                        lbl = layer.data[coord_int]
                        if lbl != 0:
                            lbls.append(lbl)
                    if len(lbls) >= 2:
                        frame_labels[t].add(tuple(lbls))

            else: # simple 3D shapes
                frame_labels.append(set())
                for polyline in shape_type_layer.data:
                    lbls = []
                    for point in polyline:
                        coord_int = tuple(int(round(x)) for x in point)
                        lbl = layer.data[coord_int]
                        if lbl != 0:
                            lbls.append(lbl)
                    if len(lbls) >= 2:
                        frame_labels[0].add(tuple(lbls))
        merged = self._merge_labels(layer.data, frame_labels)
        name = layer.name + " merged labels"
        if name in self.viewer.layers:
            self.viewer.layers[name].data = merged
        else:
            self.viewer.add_labels(merged, name=name, scale=layer.scale, units=layer.units, metadata=layer.metadata.copy())

    def kill_border_labels(self):
        layer = self.viewer.layers.selection.active
        if layer is None or not hasattr(layer, "selected_label"):
            return
        data = np.copy(layer.data if layer.ndim == 4 else layer.data[np.newaxis, ...])
        for f in range(data.shape[0]):
            data[f] = clear_border(data[f])
        name = layer.name + " no border"
        if name in self.viewer.layers:
            self.viewer.layers[name].data = data if layer.ndim == 4 and data.shape[0] > 1 else data[0]
        else:
            self.viewer.add_labels(data if layer.ndim == 4 and data.shape[0] > 1 else data[0],
                                   name=name,
                                   scale=layer.scale,
                                   units=layer.units,
                                   metadata=layer.metadata.copy())

    def props_to_dict(self, all_props, frame, use_intensities=False):
        d = {}
        for props in all_props:
            d.setdefault("Volume", []).append(props.area)
            d.setdefault("Volume AABB", []).append(props.bbox_area)
            d.setdefault("Centroid Z", []).append(props.centroid[0])
            d.setdefault("Centroid Y", []).append(props.centroid[1])
            d.setdefault("Centroid X", []).append(props.centroid[2])
            d.setdefault("AABB depth", []).append(props.bbox[3] - props.bbox[0])
            d.setdefault("AABB height", []).append(props.bbox[4] - props.bbox[1])
            d.setdefault("AABB width", []).append(props.bbox[5] - props.bbox[2])
            d.setdefault("Volume convex", []).append(props.convex_area)
            d.setdefault("Major axis length", []).append(props.major_axis_length)
            d.setdefault("Minor axis length", []).append(props.minor_axis_length)
            d.setdefault("Equivalent diameter", []).append(props.equivalent_diameter)
            d.setdefault("Solidity", []).append(props.solidity)
            d.setdefault("Label", []).append(props.label)
            d.setdefault("Frame", []).append(frame)
            if use_intensities:
                d.setdefault("Mean intensity", []).append(props.intensity_mean)
                d.setdefault("Max intensity", []).append(props.intensity_max)
                d.setdefault("Min intensity", []).append(props.intensity_min)
                d.setdefault("Integrated intensity", []).append(float(np.sum(props.intensity_image[props.intensity_image > 0])))
                d.setdefault("Intensity std. dev.", []).append(props.intensity_std)
                d.setdefault("Median intensity", []).append(float(np.median(props.intensity_image[props.intensity_image > 0])))
        return d

    def measure_labels(self):
        layer = self.viewer.layers.selection.active
        if layer is None or not hasattr(layer, "selected_label"):
            return
        intensities_name = self.intensity_channel_combobox.currentText()
        intensities = self.viewer.layers[intensities_name].data if intensities_name in self.viewer.layers else None
        if (intensities is not None) and (layer.data.shape != intensities.shape):
            return
        labels_map = layer.data if layer.ndim == 4 else layer.data[np.newaxis, ...]
        intensities_map = None if intensities is None else (intensities if intensities.ndim == 4 else intensities[np.newaxis, ...])
        all_props = {}
        scale = layer.scale if self.use_physical_units_check.isChecked() else None
        if scale is not None:
            scale = scale if layer.data.ndim == 3 else scale[1:]
        for f in range(labels_map.shape[0]):
            props = regionprops(
                labels_map[f], 
                intensity_image=None if intensities_map is None else intensities_map[f],
                spacing=scale
            )
            all_props.update(self.props_to_dict(props, f, use_intensities=intensities is not None))
        from pprint import pprint
        pprint(all_props)

def loose_launch():
    viewer = napari.Viewer()
    widget = LabelsOperations(viewer=viewer)
    viewer.window.add_dock_widget(widget)

    # import tifffile
    # data = tifffile.imread('/home/clement/Documents/formations/formation-3d-2024/images/exercise05/test-labels-frames.tif')
    # viewer.add_labels(data, name='test labels')

    napari.run()

if __name__ == "__main__":
    loose_launch()