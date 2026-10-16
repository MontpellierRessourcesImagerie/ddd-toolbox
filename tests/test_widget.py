import numpy as np

from ddd_toolbox._widget import (
    ConvertImageTypeWidget
)


def test_convert_image_type_widget(make_napari_viewer, capsys):
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()
    viewer.add_image(np.random.random((100, 100)))

    # create our widget, passing in the viewer
    convertImageTypeWidget = ConvertImageTypeWidget(viewer)

    # call our widget method
    assert convertImageTypeWidget.viewer == viewer
