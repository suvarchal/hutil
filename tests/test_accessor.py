#!/usr/bin/env python
# coding: utf-8
"""
Tests for the hutil accessor module.
"""

import pytest
import numpy as np
import xarray as xr
import healpy as hp

import hutil


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
def healpix_dataarray(healpix_dataset):
    """
    Create a HEALPix DataArray for testing.
    """
    return healpix_dataset.temperature


@pytest.fixture
def healpix_dataset_with_latlon(healpix_dataset):
    """
    Create a HEALPix dataset with latitude and longitude coordinates.
    """
    return healpix_dataset.hutil.add_latlon_coords()


@pytest.fixture
def healpix_dataarray_with_latlon(healpix_dataarray):
    """
    Create a HEALPix DataArray with latitude and longitude coordinates.
    """
    return healpix_dataarray.hutil.add_latlon_coords()


def test_dataset_accessor_registration(healpix_dataset):
    """
    Test that the dataset accessor is registered.
    """
    assert hasattr(healpix_dataset, 'hutil')
    assert isinstance(healpix_dataset.hutil, hutil.accessor.HutilDatasetAccessor)


def test_dataarray_accessor_registration(healpix_dataarray):
    """
    Test that the dataarray accessor is registered.
    """
    assert hasattr(healpix_dataarray, 'hutil')
    assert isinstance(healpix_dataarray.hutil, hutil.accessor.HutilDataArrayAccessor)


def test_get_info(healpix_dataset):
    """
    Test get_info method.
    """
    nside, nest, npix = healpix_dataset.hutil.get_info()
    assert nside == 16
    assert nest is True
    assert npix == hp.nside2npix(16)


def test_add_latlon_coords(healpix_dataset):
    """
    Test add_latlon_coords method.
    """
    ds_with_latlon = healpix_dataset.hutil.add_latlon_coords()
    assert 'lat' in ds_with_latlon.coords
    assert 'lon' in ds_with_latlon.coords
    assert len(ds_with_latlon.lat) == len(healpix_dataset.cell)
    assert len(ds_with_latlon.lon) == len(healpix_dataset.cell)


def test_select_at_latitude(healpix_dataset_with_latlon):
    """
    Test select_at_latitude method.
    """
    lat = 0.0  # Equator
    result = healpix_dataset_with_latlon.hutil.select_at_latitude(latitude=lat, tolerance=5.0)
    assert len(result.cell) > 0
    assert all(abs(result.lat.values - lat) <= 5.0)


def test_select_at_longitude(healpix_dataset_with_latlon):
    """
    Test select_at_longitude method.
    """
    lon = 0.0  # Prime meridian
    result = healpix_dataset_with_latlon.hutil.select_at_longitude(longitude=lon, tolerance=5.0)
    assert len(result.cell) > 0
    # Account for longitude wrapping
    lon_diff = np.minimum(abs(result.lon.values - lon), 360 - abs(result.lon.values - lon))
    assert all(lon_diff <= 5.0)


def test_select_at_points(healpix_dataset_with_latlon):
    """
    Test select_at_points method.
    """
    points = [(0, 0), (45, 45), (-45, -45)]
    result = healpix_dataset_with_latlon.hutil.select_at_points(points)
    assert len(result.cell) == len(points)


def test_select_region(healpix_dataset_with_latlon):
    """
    Test select_region method with bounding box.
    """
    result = healpix_dataset_with_latlon.hutil.select_region(
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
    Test select_within_polygon method.
    """
    polygon = [(0, 0), (0, 45), (45, 45), (45, 0)]
    result = healpix_dataset_with_latlon.hutil.select_within_polygon(polygon)
    assert len(result.cell) > 0
    assert all(result.lat.values >= 0)
    assert all(result.lat.values <= 45)
    assert all(result.lon.values >= 0)
    assert all(result.lon.values <= 45)


def test_get_value_at_latlon(healpix_dataset_with_latlon, healpix_dataarray_with_latlon):
    """
    Test get_value_at_latlon method.
    """
    lat, lon = 0, 0
    result = healpix_dataset_with_latlon.hutil.get_value_at_latlon(lat, lon)
    assert isinstance(result, xr.Dataset)
    
    # Test with a DataArray
    result = healpix_dataarray_with_latlon.hutil.get_value_at_latlon(lat, lon)
    assert isinstance(result, float)


def test_select_transect(healpix_dataset_with_latlon):
    """
    Test select_transect method.
    """
    lats = [0, 10, 20, 30]
    lons = [0, 10, 20, 30]
    result = healpix_dataset_with_latlon.hutil.select_transect(lats=lats, lons=lons)
    assert len(result.point) == len(lats)


def test_select(healpix_dataset_with_latlon):
    """
    Test smart select method.
    """
    # Test selecting at a latitude
    result = healpix_dataset_with_latlon.hutil.select(lat=0)
    assert len(result.cell) > 0
    
    # Test selecting at a longitude
    result = healpix_dataset_with_latlon.hutil.select(lon=0)
    assert len(result.cell) > 0
    
    # Test selecting at a point
    result = healpix_dataset_with_latlon.hutil.select(lat=0, lon=0)
    assert len(result.cell) == 1
    
    # Test selecting a region
    result = healpix_dataset_with_latlon.hutil.select(lat=slice(0, 45), lon=slice(0, 45))
    assert len(result.cell) > 0
