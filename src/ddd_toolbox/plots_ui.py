import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, 
    QHBoxLayout, QTableWidgetItem, QLabel, 
    QSlider, QPushButton, QFileDialog, QTableWidget
)
import csv
from PyQt5.QtCore import Qt
import pyqtgraph as pg


class LineProfile(QMainWindow):
    def __init__(self, profiles, name, axes=("x", "y"), parent=None):
        super().__init__(parent)

        if profiles.ndim > 2:
            raise ValueError("profiles must be a 2D array (n_frames, profile_length)")
        if profiles.ndim < 2:
            profiles = profiles[np.newaxis, :]

        self.profiles = profiles
        self.axes = axes
        self.n_frames, self.profile_len = profiles.shape
        self.exp_name = f"{name}.csv"

        self.setWindowTitle(name)
        self._init_ui()
        self._update_plot(0)

    def _init_ui(self):
        # --- Central widget + main layout ---
        self.centralWidget = QWidget(self)
        self.setCentralWidget(self.centralWidget)

        main_layout = QVBoxLayout(self.centralWidget)

        # --- Top bar: label + slider ---
        top_bar = QHBoxLayout()

        self.frame_label = QLabel(self)
        self.frame_label.setMinimumWidth(150)

        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setRange(0, self.n_frames - 1)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(1)
        self.slider.valueChanged.connect(self._on_slider_changed)

        top_bar.addWidget(self.frame_label)
        top_bar.addWidget(self.slider)

        # --- Plot area (pyqtgraph) ---
        self.plot_widget = pg.PlotWidget(self)
        self.plot_widget.setLabel("left", self.axes[1])
        self.plot_widget.setLabel("bottom", self.axes[0])
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)

        # x-axis (0..profile_len-1)
        self.x = np.arange(self.profile_len, dtype=float)
        self.curve = self.plot_widget.plot(self.x, self.profiles[0],
                                           pen=pg.mkPen(width=2))

        # --- Assemble layout ---
        main_layout.addLayout(top_bar)
        main_layout.addWidget(self.plot_widget)

        self.exportButton = QPushButton("💾 Save profile as CSV")
        self.exportButton.clicked.connect(self.export_current_profile)
        main_layout.addWidget(self.exportButton)

    def _on_slider_changed(self, value: int):
        self._update_plot(value)

    def _update_plot(self, frame_idx: int):
        frame_idx = int(frame_idx)
        if not (0 <= frame_idx < self.n_frames):
            return

        self.frame_label.setText(f"Frame: {frame_idx+1} / {self.n_frames}")
        y = self.profiles[frame_idx]
        self.curve.setData(self.x, y)

    def export_current_profile(self):    
        frame_idx = self.slider.value()
        y = self.profiles[frame_idx]
    
        fileName, _ = QFileDialog.getSaveFileName(
            self,
            "Save current profile",
            f"profile_frame_{frame_idx}.csv",
            "CSV Files (*.csv);;All Files (*)"
        )
    
        if not fileName:
            return
    
        with open(fileName, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(["index", "intensity"])
            for i, val in enumerate(y):
                writer.writerow([i, val])


class HistogramPlot(QMainWindow):
    def __init__(
        self,
        histograms: np.ndarray,
        bin_intensities: np.ndarray,
        stats: dict | None = None,
        name: str = "Histogram",
        axes=("intensity", "count"),
        parent=None,
    ):
        super().__init__(parent)

        # Normalize dims to 2D for histograms
        if histograms.ndim > 2:
            raise ValueError("histograms must be a 1D or 2D array")

        if histograms.ndim < 2:
            histograms = histograms[np.newaxis, :]

        # Normalize / broadcast bin_intensities
        if bin_intensities.ndim > 2:
            raise ValueError("bin_intensities must be a 1D or 2D array")

        if bin_intensities.ndim == 1:
            bin_intensities = np.tile(bin_intensities, (histograms.shape[0], 1))
        elif bin_intensities.ndim < 2:
            bin_intensities = bin_intensities[np.newaxis, :]

        if histograms.shape != bin_intensities.shape:
            raise ValueError(
                f"histograms shape {histograms.shape} and "
                f"bin_intensities shape {bin_intensities.shape} must match"
            )

        self.histograms = histograms
        self.bin_intensities = bin_intensities
        self.axes = axes

        self.n_frames, self.n_bins = histograms.shape
        self.exp_name = f"{name}.csv"

        # Stats handling
        self.stats = stats or {}
        # Validate stats length if provided
        for k, v in self.stats.items():
            v = np.asarray(v)
            if v.ndim != 1 or v.shape[0] != self.n_frames:
                raise ValueError(
                    f"Stats entry '{k}' must be 1D of length {self.n_frames}, got shape {v.shape}"
                )
            self.stats[k] = v  # store as array

        self.stat_keys = list(self.stats.keys())

        self.setWindowTitle(name)
        self._init_ui()
        self._update_plot(0)   # also updates stats table if present

    def _init_ui(self):
        # --- Central widget + main layout ---
        self.centralWidget = QWidget(self)
        self.setCentralWidget(self.centralWidget)

        main_layout = QVBoxLayout(self.centralWidget)

        # --- Top bar: label + slider ---
        top_bar = QHBoxLayout()

        self.frame_label = QLabel(self)
        self.frame_label.setMinimumWidth(150)

        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setRange(0, self.n_frames - 1)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(1)
        self.slider.valueChanged.connect(self._on_slider_changed)

        if self.n_frames == 1:
            self.slider.setEnabled(False)

        top_bar.addWidget(self.frame_label)
        top_bar.addWidget(self.slider)

        # --- Plot area (pyqtgraph) ---
        self.plot_widget = pg.PlotWidget(self)
        self.plot_widget.setLabel("left", self.axes[1])   # e.g. "count"
        self.plot_widget.setLabel("bottom", self.axes[0]) # e.g. "intensity"
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)

        # Initial data for frame 0
        x0 = self.bin_intensities[0]
        y0 = self.histograms[0]
        width0 = self._estimate_bin_width(x0)

        # BarGraphItem: real histogram bars with color
        self.bar_item = pg.BarGraphItem(
            x=x0,
            height=y0,
            width=width0,
            brush='steelblue',  # You can use color names or RGB tuples like (100, 150, 200)
        )
        self.plot_widget.addItem(self.bar_item)

        # --- Stats table (2 rows: headers + values) ---
        if self.stat_keys:
            self.stats_table = QTableWidget(self)
            self.stats_table.setRowCount(2)
            self.stats_table.setColumnCount(len(self.stat_keys))
            self.stats_table.verticalHeader().setVisible(False)
            self.stats_table.horizontalHeader().setVisible(False)
            self.stats_table.setEditTriggers(QTableWidget.NoEditTriggers)

            # Fill header row (row 0) with stat names
            for col, key in enumerate(self.stat_keys):
                item = QTableWidgetItem(str(key))
                self.stats_table.setItem(0, col, item)

            main_layout.addLayout(top_bar)
            main_layout.addWidget(self.plot_widget, stretch=10)
            main_layout.addWidget(self.stats_table, stretch=1)
        else:
            # No stats: just plot + top bar
            self.stats_table = None
            main_layout.addLayout(top_bar)
            main_layout.addWidget(self.plot_widget)

        # --- Export button ---
        self.exportButton = QPushButton("💾 Save histogram as CSV")
        self.exportButton.clicked.connect(self.export_current_histogram)
        main_layout.addWidget(self.exportButton)

    def _estimate_bin_width(self, x: np.ndarray) -> float:
        """Estimate bar width from bin centers."""
        if x.size <= 1:
            return 1.0
        diffs = np.diff(x)
        width = np.median(diffs)
        if not np.isfinite(width) or width <= 0:
            width = 1.0
        return width

    def _on_slider_changed(self, value: int):
        self._update_plot(value)

    def _update_plot(self, frame_idx: int):
        frame_idx = int(frame_idx)
        if not (0 <= frame_idx < self.n_frames):
            return

        self.frame_label.setText(f"Frame: {frame_idx + 1} / {self.n_frames}")

        x = self.bin_intensities[frame_idx]
        y = self.histograms[frame_idx]
        width = self._estimate_bin_width(x)

        # Update histogram bars
        self.bar_item.setOpts(x=x, height=y, width=width)

        # Update stats table if present
        if self.stats_table is not None:
            for col, key in enumerate(self.stat_keys):
                val = self.stats[key][frame_idx]
                # format value nicely; tweak as needed
                if isinstance(val, float):
                    text = f"{round(val, 4)}"
                else:
                    text = str(val)
                item = QTableWidgetItem(text)
                self.stats_table.setItem(1, col, item)

    def export_current_histogram(self):
        frame_idx = self.slider.value()
        x = self.bin_intensities[frame_idx]
        y = self.histograms[frame_idx]

        fileName, _ = QFileDialog.getSaveFileName(
            self,
            "Save current histogram",
            f"histogram_frame_{frame_idx + 1}.csv",
            "CSV Files (*.csv);;All Files (*)"
        )

        if not fileName:
            return

        with open(fileName, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=";")
            # header
            writer.writerow(["bin_center", "count"])
            for xv, yv in zip(x, y):
                writer.writerow([xv, yv])


def main_sessions():
    from qtpy.QtWidgets import QApplication
    
    frames = []
    for _ in range(10):
        dummy = np.random.randint(0, 65535, size=(1000, 1000), dtype=np.uint16)
        hist, bin_edges = np.histogram(dummy, bins=65536, range=(0, 65535))
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        frames.append((hist, bin_centers))
    
    dummy = np.array([f[0] for f in frames])
    dummy_bin_intensities = np.array([f[1] for f in frames])

    app = QApplication([])
    table = HistogramPlot(dummy, dummy_bin_intensities, "Histogram Example")
    table.show()
    app.exec_()

    print("DONE.")


if __name__ == '__main__':
    main_sessions()