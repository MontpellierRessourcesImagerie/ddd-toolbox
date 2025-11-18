"""
This module contains 3D image analysis widgets which provide a gui
for common python image analysis operations, for example from skimage
or scipy.
"""
from qtpy.QtWidgets import QWidget
from qtpy.QtWidgets import QPushButton
from qtpy.QtWidgets import QHBoxLayout
from napari.utils.events import Event

from skimage import morphology

from ddd_toolbox.lib.napari_util import NapariUtil
from ddd_toolbox.lib.qtutil import WidgetTool

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import napari



class ToolboxWidget(QWidget):


    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.field_width = 50
        self.napari_util = NapariUtil(self.viewer)
        self.image_layers = self.napari_util.getImageLayers()
        self.label_layers = self.napari_util.getLabelLayers()
        self.image_combo_boxes = []
        self.label_combo_boxes = []
        self.point_combo_boxes = []
        self.input_layer_combo_box = None
        self.label_layer_combo_box = None
        self.footprints = ["none", "cube", "ball", "octahedron"]
        self.footprint = "cube"
        self.modes = ["reflect", "constant", "nearest", "mirror", "warp"]
        self.mode = "reflect"
        self.footprint_combo_box = None
        self.footprint_radius_input = None
        self.mode_combo_box = None
        self.input_layer = None
        self.filter = None
        self.viewer.layers.events.inserted.connect(self.on_layer_added_or_removed)
        self.viewer.layers.events.removed.connect(self.on_layer_added_or_removed)


    def on_layer_added_or_removed(self, event: Event):
        self.update_layer_selection_combo_boxes()


    def update_layer_selection_combo_boxes(self):
        image_layers = self.napari_util.getImageLayers()
        label_layers = self.napari_util.getLabelLayers()
        point_layers = self.napari_util.getPointsLayers()
        for combo_box in self.image_combo_boxes:
            WidgetTool.replaceItemsInComboBox(combo_box, image_layers)
        for combo_box in self.label_combo_boxes:
            WidgetTool.replaceItemsInComboBox(combo_box, label_layers)
        for combo_box in self.point_combo_boxes:
            WidgetTool.replaceItemsInComboBox(combo_box, point_layers)


    @classmethod
    def get_footprint(cls, name, radius, dims):
        se_name = name
        two_d_ses = {'ball': 'disk', 'octahedron': 'diamond'}
        if dims == 2:
            if name in two_d_ses.keys():
                se_name = two_d_ses[name]
        if name == "cube":
            footprint_width = 2 * radius + 1
            if dims == 2:
                footprint = morphology.footprint_rectangle((footprint_width, footprint_width))
            else:
                footprint = morphology.footprint_rectangle((footprint_width, footprint_width, footprint_width))
        else:
            footprint_function = getattr(morphology, se_name)
            footprint = footprint_function(radius)
        return footprint



class ConvolutionWidget(ToolboxWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__(viewer)
        self.viewer = viewer

        btn = QPushButton("Click me!")
        btn.clicked.connect(self._on_click)

        self.setLayout(QHBoxLayout())
        self.layout().addWidget(btn)

    def _on_click(self):
        print("napari has", len(self.viewer.layers), "layers")
