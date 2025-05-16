#!/usr/bin/env python
# coding: utf-8
"""
HEALPix utilities for xarray datasets.

This package provides utilities for working with HEALPix datasets in xarray,
including selection methods for points, regions, and transects.
"""

__author__ = "Suvarchal K. Cheedela"
__email__ = "suvarchal@duck.com"
__version__ = "0.1.0"

from .selection import (
    get_healpix_info,
    add_latlon_coords,
    select_at_latitude,
    select_at_longitude,
    select_at_points,
    select_region,
    select_within_polygon,
    select_within_shapefile,
    select_transect,
    interpolate_to_grid,
    get_value_at_latlon
)

from .accessor import register_hutil_accessor

# Register the accessor when the package is imported
register_hutil_accessor()
