try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"


from ._reader import napari_get_reader
from ._sample_data import make_sample_data
import numpy as np

from ._widget import (
    ConvolutionWidget,
    FFTWidget,
    InverseFFTWidget,
    ImageCalculatorWidget,
    AddGaussianNoiseWidget,
    AddPoissonNoiseWidget,
    ImageInfoWidget,
    InvertImageWidget,
    RenameLayersWidget,
    ConvertImageTypeWidget,
)
from ._image_utils_widget import ImageUtilsWidget
from ._filters_widget import ImageFiltersWidget
from ._mask_utils_widget import MaskUtilsWidget
from ._labels_operations_widget import LabelsOperationsWidget
from ._analysis_widget import AnalysisWidget

from ._writer import write_multiple, write_single_image

__all__ = (
    "napari_get_reader",
    "write_single_image",
    "write_multiple",
    "make_sample_data",
    "ConvolutionWidget",
    "FFTWidget",
    "InverseFFTWidget",
    "ImageCalculatorWidget",
    "AddGaussianNoiseWidget",
    "AddPoissonNoiseWidget",
    "ImageInfoWidget",
    "RenameLayersWidget",
    "InvertImageWidget",
    "ConvertImageTypeWidget",
    "ImageUtilsWidget",
    "ImageFiltersWidget",
    "MaskUtilsWidget",
    "LabelsOperationsWidget",
    "AnalysisWidget"
)

np.Inf = np.inf