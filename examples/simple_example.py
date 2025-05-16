#!/usr/bin/env python
# coding: utf-8
"""
Simple example demonstrating the use of hutil package.

This script shows how to use both the direct function calls and the xarray accessor
to work with HEALPix datasets.
"""

import numpy as np
import xarray as xr
import healpy as hp
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import hutil
from hutil import selection

# Create a simple HEALPix dataset for demonstration
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

print(f'Created HEALPix dataset with nside={nside}, npix={npix}')

# Method 1: Using Direct Function Calls
print("\nMethod 1: Using Direct Function Calls")

# Get HEALPix information from the dataset
nside, nest, npix = selection.get_healpix_info(ds)
print(f'HEALPix info: nside={nside}, nest={nest}, npix={npix}')

# Add latitude and longitude coordinates to the dataset
ds_with_latlon = selection.add_latlon_coords(ds)
print('Added lat/lon coordinates to dataset')

# Select data at the equator (latitude = 0°)
equator_data = selection.select_at_latitude(ds_with_latlon, latitude=0, tolerance=2.0)
print(f'Selected {len(equator_data.cell)} points at the equator')

# Method 2: Using the xarray Accessor
print("\nMethod 2: Using the xarray Accessor")

# Get HEALPix information using the accessor
nside, nest, npix = ds.hutil.get_info()
print(f'HEALPix info: nside={nside}, nest={nest}, npix={npix}')

# Add latitude and longitude coordinates using the accessor
ds_accessor = ds.hutil.add_latlon_coords()
print('Added lat/lon coordinates to dataset using accessor')

# Select data at the equator using the accessor
equator_data = ds_accessor.hutil.select_at_latitude(latitude=0, tolerance=2.0)
print(f'Selected {len(equator_data.cell)} points at the equator using accessor')

# Plot the dataset using the accessor if matplotlib is available
print("\nPlotting the dataset...")
fig = ds_accessor.hutil.plot(title='Temperature (°C)', cmap='inferno')
plt.savefig('temperature_map.png')
print("Plot saved as 'temperature_map.png'")

# Compare the results of both methods
print("\nComparing results of both methods...")
equator_direct = selection.select_at_latitude(ds_with_latlon, latitude=0, tolerance=2.0)
equator_accessor = ds_accessor.hutil.select_at_latitude(latitude=0, tolerance=2.0)

identical = np.array_equal(equator_direct.cell.values, equator_accessor.cell.values)
print(f"Results are identical: {identical}")

print("\nExample completed successfully!")
