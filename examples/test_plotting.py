#!/usr/bin/env python
# coding: utf-8
"""
Test script to verify that the plotting functionality works correctly.
"""

import numpy as np
import xarray as xr
import healpy as hp
import matplotlib.pyplot as plt

import hutil
from hutil import selection

# Create a simple HEALPix dataset for testing
print("Creating test HEALPix dataset...")
nside = 32  # Lower resolution for faster computation
npix = hp.nside2npix(nside)
nest = True

# Create some test data - a temperature field with a hotspot
lats, lons = hp.pix2ang(nside, np.arange(npix), nest=nest, lonlat=True)

# Create a temperature field with a hotspot at (0, 0)
hotspot_lat, hotspot_lon = 0, 0
dist = np.sqrt((lats - hotspot_lat)**2 + (lons - hotspot_lon)**2)
temperature = 20 + 10 * np.exp(-dist**2 / 1000)  # Base temperature + hotspot

# Create a dataset
ds = xr.Dataset(
    data_vars={
        'temperature': ('cell', temperature),
        'humidity': ('cell', 50 + 20 * np.sin(np.radians(lats)))  # Humidity varies with latitude
    },
    coords={
        'cell': np.arange(npix)
    },
    attrs={
        'nside': nside,
        'nest': nest
    }
)

# Add latitude and longitude coordinates
ds_with_latlon = selection.add_latlon_coords(ds)
print("Added lat/lon coordinates to dataset")

# Test direct function plotting
print("\nTesting direct function plotting...")
fig = selection.plot_healpix_selection(ds_with_latlon.temperature, 
                                     title="Temperature - Direct Function", 
                                     cmap="inferno")
fig.savefig("test_direct_function.png")
print("Saved plot as 'test_direct_function.png'")

# Test accessor plotting
print("\nTesting accessor plotting...")
fig = ds_with_latlon.temperature.hutil.plot(title="Temperature - Accessor", cmap="inferno")
fig.savefig("test_accessor.png")
print("Saved plot as 'test_accessor.png'")

# Test region selection and plotting
print("\nTesting region selection and plotting...")
region_data = selection.select_region(
    ds_with_latlon,
    lat_min=-30, lat_max=30,
    lon_min=-30, lon_max=30
)
fig = selection.plot_healpix_selection(region_data.temperature, 
                                     title="Temperature in Region", 
                                     cmap="inferno")
fig.savefig("test_region.png")
print("Saved plot as 'test_region.png'")

print("\nAll tests completed successfully!")
