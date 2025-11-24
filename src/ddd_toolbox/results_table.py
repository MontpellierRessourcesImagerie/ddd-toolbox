from qtpy.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QApplication, QMainWindow,
                            QTableWidget, QTableWidgetItem, QFileDialog)
from PyQt5.QtGui import QColor, QFont
import numpy as np
import csv

class ResultsTable(QMainWindow):
    def __init__(self, data, name='Data Table', parent=None):
        super(ResultsTable, self).__init__(parent)
        self.exp_name = f"{name}.csv"
        self.setWindowTitle(name)
        self.init_ui()
        self.data_dict = data
        self.draw_table()

    def init_ui(self):
        self.centralWidget = QWidget(self)
        self.setCentralWidget(self.centralWidget)

        # Layout
        layout = QVBoxLayout(self.centralWidget)

        # Table
        self.table = QTableWidget()
        layout.addWidget(self.table)  # Add table to layout

        # Export Button
        self.exportButton = QPushButton('💾 Save as CSV')
        self.exportButton.setFont(QFont("Arial Unicode MS, Segoe UI Emoji, Apple Color Emoji, Noto Color Emoji"))
        self.exportButton.clicked.connect(self.export_data)
        layout.addWidget(self.exportButton)

    def draw_table(self):
        pass

    def set_exp_name(self, name):
        self.exp_name = ".".join(name.replace(" ", "-").split('.')[:-1]) + ".csv"

    def export_data(self):
        options = QFileDialog.Options()
        try:
            fileName, _ = QFileDialog.getSaveFileName(
                self, 
                "QFileDialog.getSaveFileName()", 
                self.exp_name,
                "CSV Files (*.csv);;All Files (*)", 
                options=options
            )
        except:
            fileName = None

        if not fileName:
            print("No file selected")
            return
        
        self.export_table_to_csv(fileName)

    def export_table_to_csv(self, filename: str):
        # Open a file in write mode
        with open(filename, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, delimiter=';')
            
            # Writing headers (optional)
            headers = [self.table.horizontalHeaderItem(i).text() if self.table.horizontalHeaderItem(i) is not None else "" for i in range(self.table.columnCount())]
            writer.writerow(headers)
            
            # Writing data
            for row in range(self.table.rowCount()):
                row_data = []
                
                # Add cell data
                for column in range(self.table.columnCount()):
                    item = self.table.item(row, column)
                    # Check if the cell is not empty
                    if item is not None:
                        row_data.append(item.text())
                    else:
                        row_data.append('')
                writer.writerow(row_data)


# ====> The first result table contains the visibility and the centroid.

def gradient_lut(n=512):
    """
    Create a custom multi-color RGBA LUT (similar vibe to viridis/fire)
    """
    # Where each color stop lies (0..1)
    stops = np.array([0.00, 0.15, 0.35, 0.55, 0.75, 1.00])

    # Color anchors (R,G,B,A)
    colors = np.array([
        [ 68,   1,  84, 255],  # purple
        [ 59,  82, 139, 255],  # deep blue
        [ 33, 145, 140, 255],  # greenish teal
        [ 94, 201,  98, 255],  # light green
        [253, 231,  37, 255],  # yellow
        [255, 108,   0, 255],  # orange-red
    ], dtype=float)

    # Allocate output
    lut = np.zeros((n, 4), dtype=np.uint8)
    x = np.linspace(0, 1, n)

    # Interpolate
    for c in range(4):
        lut[:, c] = np.interp(x, stops, colors[:, c]).astype(np.uint8)
    
    return lut

def viridis_like_lut(n=256, alpha=255):
    """
    Build a viridis-like RGBA LUT (n, 4) uint8 using linear interpolation
    between a few key colors:
        deep purple -> teal/duck -> green -> yellow
    """
    # Positions of the anchor colors along the gradient [0..1]
    stops = np.array([0.0, 0.33, 0.66, 1.0], dtype=float)

    # Anchor colors in RGBA (0–255)
    # Approximate viridis key colors:
    #   deep purple (#440154)
    #   teal/duck    (#21908C)
    #   green        (#35B779)
    #   yellow       (#FDE725)
    colors = np.array([
        [0x44, 0x01, 0x54, alpha],  # deep purple
        [0x21, 0x90, 0x8C, alpha],  # teal / duck
        [0x35, 0xB7, 0x79, alpha],  # green
        [0xFD, 0xE7, 0x25, alpha],  # yellow
    ], dtype=float)

    # Output LUT
    lut = np.zeros((n, 4), dtype=np.uint8)

    # Normalized positions in the LUT
    x = np.linspace(0.0, 1.0, n)

    # Interpolate R, G, B, A separately
    for c in range(4):
        lut[:, c] = np.clip(
            np.interp(x, stops, colors[:, c]),
            0, 255
        ).astype(np.uint8)

    return lut

def get_text_colors(lut):
    n = lut.shape[0]
    luminances = 0.2126 * lut[:, 0] + 0.7152 * lut[:, 1] + 0.0722 * lut[:, 2]
    txt_color = np.zeros((n, 4), dtype=np.uint8)
    txt_color[:, :3] = np.where(luminances[:, None] > 128, 0, 255)
    txt_color[:, 3] = 255
    return txt_color

class LabelsPropertiesResultsTable(ResultsTable):
    def __init__(self, data, name, parent=None):
        super(LabelsPropertiesResultsTable, self).__init__(data, name, parent)
        self.sorted_column = None  # Track which column is currently sorted
        self.init_ui()
        self.draw_table()
        self.table.horizontalHeader().sectionClicked.connect(self.on_header_clicked)

    def on_header_clicked(self, logical_index):
        """Handle column header clicks for sorting"""
        column_name = self.table.horizontalHeaderItem(logical_index).text()
        
        if self.sorted_column == logical_index:
            # If already sorted by this column, restore original order
            self.sorted_column = None
            self.draw_table()
        else:
            # Sort by the clicked column
            self.sorted_column = logical_index
            self.sort_by_column(column_name)

    def sort_by_column(self, column_name):
        """Sort table by the specified column"""
        # Get sort indices based on the column data
        sort_indices = np.argsort(self.data_dict[column_name])
        
        # Redraw table with sorted data
        self.draw_table(sort_indices)

    def draw_table(self, sort_indices=None):
        """Draw table with data sorted by given indices"""
        columnHeaders = list(self.data_dict.keys())
        columnHeaders.remove("Label")
        columnHeaders.remove("Frame")
        columnHeaders = ["Label", "Frame"] + columnHeaders
        
        # Sorted row headers
        rowHeaders = [str(i+1) for i in range(len(self.data_dict["Label"]))]
        if sort_indices is None:
            sort_indices = np.arange(len(self.data_dict["Label"]))
        
        self.table.setColumnCount(len(columnHeaders))
        self.table.setRowCount(len(rowHeaders))
        self.table.setHorizontalHeaderLabels(columnHeaders)
        self.table.setVerticalHeaderLabels(rowHeaders)

        # Fill table with sorted data
        lut = viridis_like_lut()
        txt_colors = get_text_colors(lut)
        for metric_idx, metric_name in enumerate(columnHeaders):
            min_val = np.min(self.data_dict[metric_name])
            max_val = np.max(self.data_dict[metric_name])
            for row_idx, data_idx in enumerate(sort_indices):
                value = self.data_dict[metric_name][data_idx]
                pos = (value - min_val) / (max_val - min_val + 1e-10)
                item = QTableWidgetItem(str(round(value, 3)))
                txt_color = QColor(*txt_colors[int(pos * (len(txt_colors) - 1))])
                color = QColor(*lut[int(pos * (len(lut) - 1))])
                item.setBackground(color)
                item.setForeground(txt_color)
                self.table.setItem(row_idx, metric_idx, item)
        
        self.table.resizeColumnsToContents()

