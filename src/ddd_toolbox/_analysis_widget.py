from qtpy.QtWidgets import (QWidget, QVBoxLayout, QTabWidget, QSpinBox,
                            QGroupBox, QHBoxLayout, QLabel, 
                            QComboBox, QCheckBox, QLineEdit, 
                            QPushButton, QFileDialog, QDoubleSpinBox,
                            QInputDialog, QDialog, QFormLayout)
from qtpy.QtCore import Qt, QThread

import napari
from napari.utils import progress
from napari.utils import Colormap

import numpy as np

from skimage.feature import peak_local_max
from skimage.morphology import h_maxima, h_minima, ball
from scipy.ndimage import map_coordinates, gaussian_filter
from skimage.measure import label, regionprops

from ddd_toolbox.plots_ui import LineProfile, HistogramPlot

NEUTRAL = "---------"

class AnalysisWidget(QWidget):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer
        self.layer_pools = []
        self.plots = []
        self.init_ui()

        self.viewer.layers.events.inserted.connect(lambda e: self.refresh_layer_names())
        self.viewer.layers.events.removed.connect(lambda e: self.refresh_layer_names())
        self.viewer.layers.events.reordered.connect(lambda e: self.refresh_layer_names())

    def init_ui(self):
        base_layout = QVBoxLayout()
        self.setLayout(base_layout)

        plotting_group = QGroupBox("Plotting")
        base_layout.addWidget(plotting_group)
        layout = QVBoxLayout()
        plotting_group.setLayout(layout)

        h_layout = QHBoxLayout()

        self.plot_line_layer_combobox = QComboBox()
        self.plot_line_layer_combobox.addItem(NEUTRAL)
        self.layer_pools.append((self.plot_line_layer_combobox, "shape_type"))
        h_layout.addWidget(self.plot_line_layer_combobox)

        self.plot_profile_button = QPushButton("Plot Line Profile")
        self.plot_profile_button.clicked.connect(self.plot_line_profile)
        h_layout.addWidget(self.plot_profile_button)
        layout.addLayout(h_layout)

        h_layout = QHBoxLayout()

        self.plot_z_layer_combobox = QComboBox()
        self.plot_z_layer_combobox.addItem(NEUTRAL)
        self.layer_pools.append((self.plot_z_layer_combobox, "shape_type"))
        h_layout.addWidget(self.plot_z_layer_combobox)

        self.plot_z_profile_button = QPushButton("Plot Z Profile")
        self.plot_z_profile_button.clicked.connect(self.plot_z_profile)
        h_layout.addWidget(self.plot_z_profile_button)
        layout.addLayout(h_layout)

        h_layout = QHBoxLayout()

        self.n_bins_spinbox = QSpinBox()
        self.n_bins_spinbox.setPrefix("Bins: ")
        self.n_bins_spinbox.setRange(2, 65536)
        self.n_bins_spinbox.setValue(256)
        h_layout.addWidget(self.n_bins_spinbox)

        self.histogram_button = QPushButton("Histogram")
        self.histogram_button.clicked.connect(self.show_histogram)
        h_layout.addWidget(self.histogram_button)
        layout.addLayout(h_layout)

        analyze_group = QGroupBox("Extrema Detection")
        base_layout.addWidget(analyze_group)
        layout = QVBoxLayout()
        analyze_group.setLayout(layout)

        h_layout = QHBoxLayout()

        self.dark_bg_checkbox = QCheckBox("Dark Background")
        self.dark_bg_checkbox.setChecked(True)
        h_layout.addWidget(self.dark_bg_checkbox)

        self.extrema_prefilter_radius_spinbox = QDoubleSpinBox()
        self.extrema_prefilter_radius_spinbox.setPrefix("Smooth: ")
        self.extrema_prefilter_radius_spinbox.setRange(0.0, 100.0)
        self.extrema_prefilter_radius_spinbox.setValue(0.0)
        h_layout.addWidget(self.extrema_prefilter_radius_spinbox)
        layout.addLayout(h_layout)

        h_layout = QHBoxLayout()

        self.prominence_spinbox = QDoubleSpinBox()
        self.prominence_spinbox.setPrefix("Prominence: ")
        self.prominence_spinbox.setRange(0.0, 100000.0)
        self.prominence_spinbox.setValue(10.0)
        self.prominence_spinbox.setDecimals(4)
        h_layout.addWidget(self.prominence_spinbox)
        

        self.find_extrema_button = QPushButton("Find Extrema")
        self.find_extrema_button.clicked.connect(self.find_extrema)
        h_layout.addWidget(self.find_extrema_button)

        layout.addLayout(h_layout)
    
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

    def plot_line_profile(self):
        intensities_layer = self.viewer.layers.selection.active
        if intensities_layer is None or not hasattr(intensities_layer, 'colormap'):
            print("No active layer selected")
            return
        intensities_data = intensities_layer.data if intensities_layer.data.ndim == 4 else intensities_layer.data[np.newaxis, :]
        line_name = self.plot_line_layer_combobox.currentText()
        if line_name not in self.viewer.layers:
            print("No line layer selected")
            return
        line_layer = self.viewer.layers[line_name]
        if line_layer.ndim != intensities_layer.ndim:
            print("Line layer and intensities layer have different dimensions")
            return
        true_len = min(len(line_layer.data), len(line_layer.shape_type))
        if true_len != 1:
            print("The shape layer must contain exactly one line or path")
            return
        if line_layer.shape_type[true_len-1] not in ['line', 'path']:
            print("Selected layer is not a line or path")
            return
        polyline = line_layer.data[true_len-1]
        all_samples = np.empty((0, 3), dtype=float)
        for i in range(len(polyline)-1):
            p0 = polyline[i][-3:]
            p1 = polyline[i+1][-3:]
            segment_len = np.linalg.norm(p1 - p0)
            n_samples = int(np.ceil(segment_len))
            if n_samples < 2:
                n_samples = 2
            samples = np.linspace(p0, p1, n_samples, endpoint=False)
            all_samples = np.vstack((all_samples, samples))
        
        metrics = np.zeros((intensities_data.shape[0], all_samples.shape[0]), dtype=float)
        for frame_idx in range(intensities_data.shape[0]):
            sampled_values = map_coordinates(intensities_data[frame_idx], all_samples.T, order=1, mode='nearest')
            metrics[frame_idx, :] = sampled_values
        name = f"Line profile {intensities_layer.name}"
        index = 1
        while name in self.viewer.layers:
            name = f"Line profile {intensities_layer.name} ({index})"
            index += 1
        profile_plot = LineProfile(metrics, name=name, axes=("Position along line", "Intensity"), parent=self)
        self.plots.append(profile_plot)
        profile_plot.show()

    def plot_z_profile(self):
        intensity_layer = self.viewer.layers.selection.active
        if intensity_layer is None or not hasattr(intensity_layer, 'colormap'):
            print("No active layer selected")
            return
        intensity_data = intensity_layer.data if intensity_layer.data.ndim == 4 else intensity_layer.data[np.newaxis, :]
        shape_name = self.plot_z_layer_combobox.currentText()
        if shape_name not in self.viewer.layers:
            print("No shape layer selected")
            return
        shape_layer = self.viewer.layers[shape_name]
        if shape_layer.ndim != intensity_layer.ndim:
            print("Shape layer and intensity layer have different dimensions")
            return
        true_len = min(len(shape_layer.data), len(shape_layer.shape_type))
        if true_len != 1:
            print("The shape layer must contain exactly one shape")
            return
        shape_type = shape_layer.shape_type[true_len-1]
        if shape_type not in ['rectangle']:
            print("Selected layer is not a rectangle")
            return
        rect = shape_layer.data[true_len-1].T
        min_y, min_x = np.min(rect[-2]), np.min(rect[-1])
        max_y, max_x = np.max(rect[-2]), np.max(rect[-1])
        z_profiles = np.zeros((intensity_data.shape[0], intensity_data.shape[1]), dtype=float)
        for frame_idx in range(intensity_data.shape[0]):
            for z in range(intensity_data.shape[1]):
                slice_2d = intensity_data[frame_idx, z]
                region = slice_2d[int(min_y):int(max_y)+1, int(min_x):int(max_x)+1]
                z_profiles[frame_idx, z] = np.mean(region)
        name = f"Z profile {intensity_layer.name}"
        index = 1
        while name in self.viewer.layers:
            name = f"Z profile {intensity_layer.name} ({index})"
            index += 1
        profile_plot = LineProfile(z_profiles, name=name, axes=("Z slice index", "Mean intensity"), parent=self)
        self.plots.append(profile_plot)
        profile_plot.show()


    def show_histogram(self):
        layer = self.viewer.layers.selection.active
        if layer is None or not hasattr(layer, 'colormap'):
            print("No active layer selected")
            return
        full_data = layer.data if layer.data.ndim == 4 else layer.data[np.newaxis, :]
        n_bins = self.n_bins_spinbox.value()
        histograms = []
        bin_centers = None
        frame_stats = {'min': [], 'max': [], 'mean': [], 'std': [], 'median': []}
        for frame_idx in range(full_data.shape[0]):
            frame_data = full_data[frame_idx]
            frame_stats['min'].append(np.min(frame_data))
            frame_stats['max'].append(np.max(frame_data))
            frame_stats['mean'].append(np.mean(frame_data))
            frame_stats['std'].append(np.std(frame_data))
            frame_stats['median'].append(np.median(frame_data))
            hist, bin_edges = np.histogram(frame_data, bins=n_bins, range=(np.min(frame_data), np.max(frame_data)))
            histograms.append(hist)
            if bin_centers is None:
                bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        histograms = np.array(histograms)
        bin_centers = np.array(bin_centers)
        name = f"Histogram {layer.name}"
        index = 1
        while name in self.viewer.layers:
            name = f"Histogram {layer.name} ({index})"
            index += 1
        histogram_plot = HistogramPlot(histograms, bin_centers, frame_stats, name=name, axes=("Intensity", "Count"), parent=self)
        self.plots.append(histogram_plot)
        histogram_plot.show()
    
    def find_extrema(self):
        layer = self.viewer.layers.selection.active
        if layer is None or not hasattr(layer, 'colormap'):
            print("No active layer selected")
            return
        full_data = layer.data if layer.data.ndim == 4 else layer.data[np.newaxis, :]
        prominence = self.prominence_spinbox.value()
        buffer = []
        found = 0
        for frame_idx in range(full_data.shape[0]):
            frame_data = full_data[frame_idx]
            z, _, x = layer.scale[-3:]
            anisotropy = x / z
            data = frame_data.astype(float)
            sigma = self.extrema_prefilter_radius_spinbox.value()
            if sigma > 0.0:
                sigmas = [sigma * anisotropy, sigma, sigma]
                data = gaussian_filter(data, sigma=sigmas)
            
            extrema_finder = h_maxima if self.dark_bg_checkbox.isChecked() else h_minima
            hmax_mask = extrema_finder(data, h=prominence)
            labels = label(hmax_mask)
            coords = []
            for rp in regionprops(labels):
                coords.append(rp.centroid)
            coords = np.array(coords)
            found += len(coords)
            if layer.ndim == 4:
                coords = np.hstack([np.full((len(coords), 1), frame_idx), coords])
            buffer.append(coords)
        coords = np.vstack(buffer)
        if found == 0:
            print("No extrema found")
            return
        name = f"{layer.name} Extrema"
        if name in self.viewer.layers:
            self.viewer.layers[name].data = coords
        else:
            self.viewer.add_points(coords, name=name, size=5, face_color='red', scale=layer.scale, units=layer.units)


def loose_launch():
    viewer = napari.Viewer()
    widget = AnalysisWidget(viewer=viewer)
    viewer.window.add_dock_widget(widget)

    # import tifffile
    # image = tifffile.imread('/home/clement/Documents/formations/formation-3d-2024/images/exercise05/SPOT-Zebra LoG 1.tif')
    # l = viewer.add_image(image, name='nuclei')
    # l.scale = (2.01, 1.2760, 1.2760)

    napari.run()

if __name__ == "__main__":
    loose_launch()