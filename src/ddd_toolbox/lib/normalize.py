import numpy as np
from ddd_toolbox.lib.axes import AxesUtils

class NormalizeValues(object):

    """
    Stretches the values of an image to a target range.
    The range can be specified manually or determined automatically based on the output type.
    If per_frame is True, normalization is done independently for each frame (min and max reprocessed for each frame). 
    Otherwise, the global min and max of the entire image are used.
    """

    def __init__(self, image: np.ndarray, type_out: str, per_frame: bool, bounds: tuple[float | None, float | None]):
        self.image = AxesUtils.as4d(image)
        self.type_out = type_out
        self.per_frame = per_frame
        self.result = None
        self.bounds = bounds

    @classmethod
    def known_types(cls):
        return ["uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "float32", "float64"]

    def auto_bounds(self, dtype: np.dtype) -> tuple[float, float]:
        if dtype == np.uint8:
            return (0, 255)
        elif dtype == np.int8:
            return (-128, 127)
        elif dtype == np.uint16:
            return (0, 65535)
        elif dtype == np.int16:
            return (-32768, 32767)
        elif dtype == np.uint32:
            return (0, 4294967295)
        elif dtype == np.int32:
            return (-2147483648, 2147483647)
        elif dtype == np.uint64:
            return (0, 18446744073709551615)
        elif dtype == np.int64:
            return (-9223372036854775808, 9223372036854775807)
        elif dtype == np.float32 or dtype == np.float64:
            return (0.0, 1.0)
        else:
            raise ValueError("Unsupported output type for auto bounds.")
        
    def stretch_image(self, image: np.ndarray, img_min: float | None, img_max: float | None, target_min: float, target_max: float, dtype: np.dtype) -> np.ndarray:
        low = np.min(image) if img_min is None else img_min
        high = np.max(image) if img_max is None else img_max
        if np.abs(high - low) < 1e-6:
            return np.zeros(image.shape, dtype=dtype)
        tmp = np.copy(image).astype(np.float32) if image.dtype not in set([np.float32, np.float64]) else np.copy(image)
        scaled = (tmp - low) / (high - low)
        stretched = scaled * (target_max - target_min) + target_min
        return stretched.astype(dtype)
    
    def run(self):
        dtype = np.dtype(self.type_out)
        t_min, t_max = self.auto_bounds(dtype)
        if self.bounds[0] is not None:
            t_min = self.bounds[0]
        if self.bounds[1] is not None:
            t_max = self.bounds[1]
        self.bounds = (t_min, t_max)
        img_min = np.min(self.image)
        img_max = np.max(self.image)
        transformed = []
        for frame_idx in range(self.image.shape[0]):
            frame = self.image[frame_idx]
            if self.per_frame:
                img_min = np.min(frame)
                img_max = np.max(frame)
            normalized_frame = self.stretch_image(frame, img_min, img_max, t_min, t_max, dtype)
            transformed.append(normalized_frame)
        self.result = AxesUtils.merge_frames(transformed)