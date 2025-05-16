#!/usr/bin/env python
# coding: utf-8
"""
Example demonstrating the use of hutil package with the Digital Earth Hackathon catalog.

This script shows how to load data from the Digital Earth Hackathon catalog and use
the hutil package to perform selection operations on HEALPix data.

Author: Suvarchal K. Cheedela
"""

import intake
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import warnings

# Import hutil
import hutil
from hutil import selection

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Load the catalog
print("Loading the Digital Earth Hackathon catalog...")
current_location = "online"
cat = intake.open_catalog("https://digital-earths-global-hackathon.github.io/catalog/catalog.yaml")[current_location]

# List available datasets
print("\nAvailable datasets in the catalog:")
print(list(cat))

# Select a dataset (ICON model)
dataset_name = "icon_d3hp003"
print(f"\nLoading {dataset_name} dataset with zoom level 7 (approx. 50km resolution)...")

# Load the dataset with zoom level 7 (approx. 50km resolution)
ds = cat[dataset_name](zoom=7).to_dask()

# Print dataset information
print(f"\nDataset information:")
print(f"  - Dimensions: {dict(ds.dims)}")
print(f"  - Variables: {list(ds.data_vars)}")
print(f"  - Time range: {ds.time.values[0]} to {ds.time.values[-1]}")

# Select a variable and time step
var_name = "ts"  # Surface temperature
time_idx = 0  # First time step

print(f"\nSelecting variable '{var_name}' at time {ds.time.values[time_idx]}...")
da = ds[var_name].isel(time=time_idx)

# Method 1: Using Direct Function Calls
print("\nMethod 1: Using Direct Function Calls")

# Get HEALPix information from the dataset
nside, nest, npix = selection.get_healpix_info(da)
print(f"HEALPix info: nside={nside}, nest={nest}, npix={npix}")

# Add latitude and longitude coordinates to the dataset
da_with_latlon = selection.add_latlon_coords(da)
print("Added lat/lon coordinates to dataset")

# Plot the original dataset
plt.figure(figsize=(12, 8))
selection.plot_healpix_selection(da_with_latlon, 
                                title=f"{var_name} at {ds.time.values[time_idx]}", 
                                cmap="inferno")
plt.savefig("catalog_original.png")
print("Saved plot as 'catalog_original.png'")

# Select data at the equator (latitude = 0°)
equator_data = selection.select_at_latitude(da_with_latlon, latitude=0, tolerance=2.0)
print(f"Selected {len(equator_data.cell)} points at the equator")

# Plot the selected data
plt.figure(figsize=(12, 8))
selection.plot_healpix_selection(equator_data, 
                                title=f"{var_name} at Equator", 
                                cmap="inferno")
plt.savefig("catalog_equator.png")
print("Saved plot as 'catalog_equator.png'")

# Select a region (Africa)
print("\nSelecting data in the Africa region...")
africa_data = selection.select_region(
    da_with_latlon,
    lat_min=-35, lat_max=35,
    lon_min=-20, lon_max=55
)
print(f"Selected {len(africa_data.cell)} points in the Africa region")

# Plot the selected data
plt.figure(figsize=(12, 8))
selection.plot_healpix_selection(africa_data, 
                                title=f"{var_name} in Africa", 
                                cmap="inferno")
plt.savefig("catalog_africa.png")
print("Saved plot as 'catalog_africa.png'")

# Method 2: Using the xarray Accessor
print("\nMethod 2: Using the xarray Accessor")

# Get HEALPix information using the accessor
nside, nest, npix = da.hutil.get_info()
print(f"HEALPix info: nside={nside}, nest={nest}, npix={npix}")

# Add latitude and longitude coordinates using the accessor
da_accessor = da.hutil.add_latlon_coords()
print("Added lat/lon coordinates to dataset using accessor")

# Select data at the equator using the accessor
equator_data_accessor = da_accessor.hutil.select_at_latitude(latitude=0, tolerance=2.0)
print(f"Selected {len(equator_data_accessor.cell)} points at the equator using accessor")

# Select a region (South America) using the accessor
print("\nSelecting data in the South America region using accessor...")
sam_data = da_accessor.hutil.select_region(
    lat_min=-55, lat_max=15,
    lon_min=-85, lon_max=-30
)
print(f"Selected {len(sam_data.cell)} points in the South America region")

# Plot the selected data using the accessor
plt.figure(figsize=(12, 8))
sam_data.hutil.plot(title=f"{var_name} in South America - Accessor", cmap="inferno")
plt.savefig("catalog_southamerica.png")
print("Saved plot as 'catalog_southamerica.png'")

# Interpolate to a regular grid
print("\nInterpolating to a regular grid...")
grid_data = da_accessor.hutil.interpolate_to_grid(
    lat_res=1.0, lon_res=1.0,
    lat_min=-90, lat_max=90, lon_min=-180, lon_max=180,
    method="nearest"
)

# Plot the interpolated data
plt.figure(figsize=(12, 8))
ax = plt.axes(projection=ccrs.Robinson())
grid_data.plot(ax=ax, transform=ccrs.PlateCarree(), cmap="inferno")
ax.coastlines()
ax.gridlines(draw_labels=True)
plt.title(f"Interpolated {var_name} - Global View")
plt.savefig("catalog_interpolated.png")
print("Saved plot as 'catalog_interpolated.png'")

print("\nExample completed successfully!")
