"""
This module is an example of a barebones numpy reader plugin for napari.

It implements the Reader specification, but your plugin may choose to
implement multiple readers or even other plugin contributions. see:
https://napari.org/stable/plugins/building_a_plugin/guides.html#readers
"""

import numpy as np
from tifffile import TiffFile
import napari


def napari_get_reader(path):
    """A basic implementation of a Reader contribution.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    function or None
        If the path is a recognized format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """
    if isinstance(path, list):
        # reader plugins may be handed single path, or a list of paths.
        # if it is a list, it is assumed to be an image stack...
        # so we are only going to look at the first file.
        path = path[0]

    # if we know we cannot read the file, we immediately return None.
    extension = path.split(".")[-1].lower()
    if not extension in ["tif", "tiff"]:
        return None

    # otherwise we return the *function* that can read ``path``.
    return reader_function


def reader_function(path):
    """Take a path or list of paths and return a list of LayerData tuples.

    Readers are expected to return data as a list of tuples, where each tuple
    is (data, [add_kwargs, [layer_type]]), "add_kwargs" and "layer_type" are
    both optional.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    layer_data : list of tuples
        A list of LayerData tuples where each tuple in the list contains
        (data, metadata, layer_type), where data is a numpy array, metadata is
        a dict of keyword arguments for the corresponding viewer.add_* method
        in napari, and layer_type is a lower-case string naming the type of
        layer. Both "meta", and "layer_type" are optional. napari will
        default to layer_type=="image" if not provided
    """
    # handle both a string and a list of strings
    paths = [path] if isinstance(path, str) else path
    # load all files into array
    resultTriples = []
    for _path in paths:
        with TiffFile(_path) as tif:
            volume = tif.asarray()
            axes = tif.series[0].axes
            voxelSizeX = 1
            voxelSizeY = 1
            if 282 in tif.pages[0].tags.keys():
                voxelSizeX = tif.pages[0].tags['XResolution'].value[1] / tif.pages[0].tags['XResolution'].value[0]
            if 283 in tif.pages[0].tags.keys():
                voxelSizeY = tif.pages[0].tags['YResolution'].value[1] / tif.pages[0].tags['XResolution'].value[0]
            spacing = 1
            if 'spacing' in tif.imagej_metadata.keys():
                spacing = tif.imagej_metadata['spacing']
            unit = 'pixel'
            if 'unit' in tif.imagej_metadata.keys():
                unit = tif.imagej_metadata['unit']
                if unit == '\\u00B5m':
                    unit = 'micrometer'
        add_kwargs = {}
        if 'C' in axes:
            channel_axis = axes.index('C')
            add_kwargs["channel_axis"] = channel_axis
        add_kwargs['blending'] = 'additive'
        add_kwargs['depiction'] = 'volume'
        if 'Z' in axes:
            add_kwargs['scale'] = (spacing, voxelSizeY, voxelSizeX)
        else:
            add_kwargs['scale'] = (voxelSizeY, voxelSizeX)
        add_kwargs['units'] = unit
        layer_type = "image"  # optional, default is "image"
        resultTriples.append((volume, add_kwargs, layer_type))
    if 'Z' in axes:
        napari.viewer.current_viewer().dims.ndisplay = 3
    return resultTriples
