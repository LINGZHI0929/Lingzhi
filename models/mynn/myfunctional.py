"""Functional interface"""

from . import _myfunctions


def interp_bilinear(input, size=None, zoom_factor=None, shrink_factor=None):
    return _myfunctions.InterpBilinear2d(size, zoom_factor, shrink_factor)(input)
