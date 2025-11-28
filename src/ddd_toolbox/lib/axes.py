import numpy as np

class AxesUtils:

    @classmethod
    def get_oriented_spatial_axes(cls):
        return set(['+Z', '-Z', '+Y', '-Y', '+X', '-X'])
    
    @classmethod
    def get_spatial_axes(cls):
        return set([a for a in cls.get_oriented_spatial_axes() if a.replace('+', '').replace('-', '')])

    @classmethod
    def as4d(cls, array: np.ndarray) -> np.ndarray:
        if array.ndim == 4:
            return np.copy(array)
        elif array.ndim == 3:
            return array[np.newaxis, ...]
        elif array.ndim <= 2:
            raise ValueError("Input array must have at least 3 dimensions.")
        else:
            raise ValueError("Input array must have at most 4 dimensions (TZYX).")
    
    @classmethod
    def merge_frames(cls, images: list[np.ndarray]) -> np.ndarray:
        return np.squeeze(np.stack(images, axis=0))


class SplitAxes(object):

    def __init__(self, image: np.ndarray, scale: tuple, units: tuple, axis_index: int):
        self.image = image
        self.scale = scale
        self.units = units
        self.axis_index = axis_index
        self.result = []

    def pop_item(self, elements: tuple):
        return tuple([e for i, e in enumerate(elements) if i != self.axis_index])
    
    def pop_scale(self):
        return self.pop_item(self.scale)
    
    def pop_units(self):
        return self.pop_item(self.units)

    def dimension_check(self):
        if self.axis_index < 0 or self.axis_index >= self.image.ndim:
            raise ValueError("Axis index is out of bounds for the input image.")

    def run(self):
        self.dimension_check()
        self.result = np.split(self.image, self.image.shape[self.axis_index], axis=self.axis_index)
        self.result = [np.squeeze(img) for img in self.result]
        self.scale = self.pop_scale()
        self.units = self.pop_units()


class Reslice(object):
    
    def __init__(self, image: np.ndarray, direction: str, scale: list, units: list):
        self.data_dim = image.ndim
        self.image = AxesUtils.as4d(image)
        self.direction = direction
        self.scale = scale
        self.units = units
        self.result = None
        self.result_scale = self.scale.copy()
        self.result_units = self.units.copy()

    def run(self):
        if self.direction not in AxesUtils.get_oriented_spatial_axes():
            raise ValueError(f"Direction must be one of: {', '.join(AxesUtils.get_oriented_spatial_axes())}")
        dims = set([self.data_dim, len(self.scale), len(self.units)])
        if len(dims) != 1:
            raise ValueError("Image, scale, and units must have matching dimensions.")
        self.result = self.reslice()
        self.update_scale()
        self.update_units()

    def _update_order(self, array_in, array_out):
        z_ax, y_ax, x_ax = -3, -2, -1
        if self.direction in ["+Z", "-Z"]:
            pass
        elif self.direction in ["+Y", "-Y"]:
            array_out[z_ax] = array_in[y_ax]
            # New Y scale is old Z scale
            array_out[y_ax] = array_in[z_ax]
        elif self.direction in ["+X", "-X"]:
            # New Z scale is old X scale
            array_out[z_ax] = array_in[x_ax]
            # New X scale is old Z scale
            array_out[x_ax] = array_in[z_ax]

    def update_scale(self):
        self._update_order(self.scale, self.result_scale)

    def update_units(self):
        self._update_order(self.units, self.result_units)

    def _view_through_axis(self, img: np.ndarray) -> np.ndarray:
        if img.ndim != 3:
            raise ValueError("Input image must be 3D (Z, Y, X), not 3D+t.")

        if self.direction == "+Z":
            return img
        elif self.direction == "-Z":
            return img[::-1, :, :]
        elif self.direction == "+Y":
            out = np.swapaxes(img, 0, 1) # (Y, Z, X)
            return out
        elif self.direction == "-Y":
            out = np.swapaxes(img, 0, 1) # (Y, Z, X)
            return out[::-1, :, :]
        elif self.direction == "+X":
            out = np.swapaxes(img, 0, 2) # (X, Y, Z)
            return out
        else:  # self.direction == "-X"
            out = np.swapaxes(img, 0, 2) # (X, Y, Z)
            return out[::-1, :, :]
    
    def reslice(self) -> np.ndarray:
        transformed = []
        for frame_idx in range(self.image.shape[0]):
            v = self.image[frame_idx]
            t = self._view_through_axis(v)
            transformed.append(t)
        return AxesUtils.merge_frames(transformed)
