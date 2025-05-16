#!/usr/bin/env python
# coding: utf-8
"""
Tests for the hutil selection module.
"""

import os
import pytest
import numpy as np
import xarray as xr
import healpy as hp
from pathlib import Path

from hutil import selection
from hutil.selection import (
    get_healpix_info,
    add_latlon_coords,
    select_at_latitude,
    select_at_longitude,
    select_at_points,
    select_region,
    select_within_polygon,
    get_value_at_latlon,
    select_transect,
    select
)


@pytest.fixture
def healpix_dataset():
    """
    Create a simple HEALPix dataset for testing.
    """
    nside = 16
    npix = hp.nside2npix(nside)
    nest = True
    
    # Create some test data
    data = np.arange(npix, dtype=float)
    
    # Create a dataset
    ds = xr.Dataset(
        data_vars={
            'temperature': ('cell', data),
            'humidity': ('cell', data * 0.1)
        },
        coords={
            'cell': np.arange(npix)
        },
        attrs={
            'nside': nside,
            'nest': nest
        }
    )
    
    return ds


@pytest.fixture
def healpix_dataset_with_latlon(healpix_dataset):
    """
    Create a HEALPix dataset with latitude and longitude coordinates.
    """
    return add_latlon_coords(healpix_dataset)


def test_get_healpix_info(healpix_dataset):
    """
    Test get_healpix_info function.
    """
    nside, nest, npix = get_healpix_info(healpix_dataset)
    assert nside == 16
    assert nest is True
    assert npix == hp.nside2npix(16)


def test_add_latlon_coords(healpix_dataset, healpix_dataset_with_latlon):
    """
    Test add_latlon_coords function.
    """
    assert 'lat' in healpix_dataset_with_latlon.coords
    assert 'lon' in healpix_dataset_with_latlon.coords
    assert len(healpix_dataset_with_latlon.lat) == len(healpix_dataset.cell)
    assert len(healpix_dataset_with_latlon.lon) == len(healpix_dataset.cell)


def test_select_at_latitude(healpix_dataset_with_latlon):
    """
    Test select_at_latitude function.
    """
    lat = 0.0  # Equator
    result = select_at_latitude(healpix_dataset_with_latlon, latitude=lat, tolerance=5.0)
    assert len(result.cell) > 0
    assert all(abs(result.lat.values - lat) <= 5.0)


def test_select_at_longitude(healpix_dataset_with_latlon):
    """
    Test select_at_longitude function.
    """
    lon = 0.0  # Prime meridian
    result = select_at_longitude(healpix_dataset_with_latlon, longitude=lon, tolerance=5.0)
    assert len(result.cell) > 0
    # Account for longitude wrapping
    lon_diff = np.minimum(abs(result.lon.values - lon), 360 - abs(result.lon.values - lon))
    assert all(lon_diff <= 5.0)


def test_select_at_points(healpix_dataset_with_latlon):
    """
    Test select_at_points function.
    """
    points = [(0, 0), (45, 45), (-45, -45)]
    result = select_at_points(healpix_dataset_with_latlon, points)
    assert len(result.cell) == len(points)


def test_select_region(healpix_dataset_with_latlon):
    """
    Test select_region function with bounding box.
    """
    result = select_region(
        healpix_dataset_with_latlon,
        lat_min=0, lat_max=45,
        lon_min=0, lon_max=45
    )
    assert len(result.cell) > 0
    assert all(result.lat.values >= 0)
    assert all(result.lat.values <= 45)
    assert all(result.lon.values >= 0)
    assert all(result.lon.values <= 45)


def test_select_within_polygon(healpix_dataset_with_latlon):
    """
    Test select_within_polygon function.
    """
    polygon = [(0, 0), (0, 45), (45, 45), (45, 0)]
    result = select_within_polygon(healpix_dataset_with_latlon, polygon)
    assert len(result.cell) > 0
    assert all(result.lat.values >= 0)
    assert all(result.lat.values <= 45)
    assert all(result.lon.values >= 0)
    assert all(result.lon.values <= 45)


def test_get_value_at_latlon(healpix_dataset_with_latlon):
    """
    Test get_value_at_latlon function.
    """
    lat, lon = 0, 0
    result = get_value_at_latlon(healpix_dataset_with_latlon, lat, lon)
    assert isinstance(result, xr.Dataset)
    
    # Test with a DataArray
    result = get_value_at_latlon(healpix_dataset_with_latlon.temperature, lat, lon)
    assert isinstance(result, float)


def test_select_transect(healpix_dataset_with_latlon):
    """
    Test select_transect function.
    """
    lats = [0, 10, 20, 30]
    lons = [0, 10, 20, 30]
    result = select_transect(healpix_dataset_with_latlon, lats=lats, lons=lons)
    assert len(result.point) == len(lats)


def test_select(healpix_dataset_with_latlon):
    """
    Test smart select function.
    """
    # Test selecting at a latitude
    result = select(healpix_dataset_with_latlon, lat=0)
    assert len(result.cell) > 0
    
    # Test selecting at a longitude
    result = select(healpix_dataset_with_latlon, lon=0)
    assert len(result.cell) > 0
    
    # Test selecting at a point
    result = select(healpix_dataset_with_latlon, lat=0, lon=0)
    assert len(result.cell) == 1
    
    # Test selecting a region
    result = select(healpix_dataset_with_latlon, lat=slice(0, 45), lon=slice(0, 45))
    assert len(result.cell) > 0
