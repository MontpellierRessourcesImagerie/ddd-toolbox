from qtpy.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QApplication, QMainWindow,
                            QTableWidget, QTableWidgetItem, QFileDialog)
from PyQt5.QtGui import QColor, QFont
import numpy as np
import csv
import math

class ResultsTable(QMainWindow):
    def __init__(self, data, name='Data Table', parent=None):
        super(ResultsTable, self).__init__(parent)
        self.exp_name = "untitled.csv"
        self.setWindowTitle(name)
        self.init_ui()
        self.data_dict = data
        self.draw_table()

    def init_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

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
                for column in range(self.table.columnCount()):
                    item = self.table.item(row, column)
                    # Check if the cell is not empty
                    if item is not None:
                        row_data.append(item.text())
                    else:
                        row_data.append('')
                writer.writerow(row_data)


# ====> The first result table contains the visibility and the centroid.


class LabelsPropertiesResultsTable(ResultsTable):
    def __init__(self, data, name, parent=None):
        super(LabelsPropertiesResultsTable, self).__init__(data, name, parent)
        self.init_ui()
        self.draw_table()

    def draw_table(self):
        columnHeaders = list(self.data_dict.keys())
        columnHeaders.remove("Label") # Labels are row headers
        columnHeaders.remove("Frame") # Put 'Frame' first
        columnHeaders = ["Frame"] + columnHeaders
        nHeaders = len(columnHeaders)
        
        # Settings rows headers.
        rowHeaders = [str(i) for i in self.data_dict["Label"]]
        
        self.table.setColumnCount(len(columnHeaders))
        self.table.setRowCount(len(rowHeaders))
        self.table.setHorizontalHeaderLabels(columnHeaders)
        self.table.setVerticalHeaderLabels(rowHeaders)

        # Filling the table.
        for metric_idx, metric_name in enumerate(columnHeaders): # column index
            for line_idx, value in enumerate(self.data_dict[metric_name]): # row index
                item = QTableWidgetItem(str(value))
                color = QColor(255, 255, 255, 100)
                item.setBackground(color)
                self.table.setItem(metric_idx, line_idx, item)
        
        self.table.resizeColumnsToContents()
