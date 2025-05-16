#!/usr/bin/env python
# coding: utf-8
"""
Enhanced plotting examples for HEALPix data.

This script demonstrates the enhanced plotting capabilities added to the hutil package,
including improved map plots and time-space cross-section plots.

Acknowledgments:
These plotting functions are inspired by and adapted from the EasyGEMS HEALPix
visualization examples available at:
- https://easy.gems.dkrz.de/Processing/healpix/healpix_cartopy.html
- https://easy.gems.dkrz.de/Processing/healpix/time-space.html
"""

import numpy as np
import xarray as xr
import healpy as hp
import matplotlib.pyplot as plt
import datetime as dt

import hutil
from hutil import plotting

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

# Add some time dimension for time-space plots
num_times = 24
times = [dt.datetime(2023, 1, 1) + dt.timedelta(hours=i) for i in range(num_times)]

# Create a dataset with time dimension
time_data = np.zeros((num_times, npix))

# Create a wave pattern that moves over time
for i, time in enumerate(times):
    # Create a wave pattern that moves with time
    wave_center_lon = (i * 15) % 360  # Move 15 degrees per hour
    dist_to_wave = np.minimum(
        np.abs(lons - wave_center_lon),
        np.abs(lons - wave_center_lon + 360),
        np.abs(lons - wave_center_lon - 360)
    )
    wave_pattern = 5 * np.exp(-dist_to_wave**2 / 500) * np.cos(np.radians(lats) * 3)
    
    # Add the wave pattern to the base temperature
    time_data[i, :] = temperature + wave_pattern

# Create a dataset with time dimension
ds_time = xr.Dataset(
    data_vars={
        'temperature': (['time', 'cell'], time_data)
    },
    coords={
        'time': times,
        'cell': np.arange(npix)
    },
    attrs={
        'nside': nside,
        'nest': nest
    }
)

# Add latitude and longitude coordinates
ds_time = hutil.selection.add_latlon_coords(ds_time)
print("Created dataset with time dimension")

# Example 1: Enhanced map plot using different projections
print("\nExample 1: Enhanced map plots with different projections")
projections = ['PlateCarree', 'Robinson', 'Orthographic', 'Mollweide']

# We'll create each plot separately to handle any projection-specific issues
for proj in projections:
    # Get a single time slice
    data_slice = ds_time.temperature.isel(time=0)
    
    # Set central longitude based on projection
    if proj == 'Orthographic':
        central_lon = 0
    else:
        central_lon = 0
    
    try:
        # Create enhanced map plot with error handling
        fig_map = plotting.plot_map(
            data_slice,
            projection=proj,
            cmap='viridis',
            title=f'{proj} Projection',
            central_longitude=central_lon,
            robust=True,
            figsize=(8, 6)
        )
        plt.close(fig_map)  # Close the figure to avoid displaying it
        # Save the figure
        fig_map.savefig(f"enhanced_map_{proj.lower()}.png")
        print(f"Saved map with {proj} projection as 'enhanced_map_{proj.lower()}.png'")
    except Exception as e:
        print(f"Error creating {proj} projection: {e}")
    


# Example 2: Time-latitude diagram
print("\nExample 2: Time-latitude diagram")
try:
    fig_time_lat = plotting.plot_time_latitude(
        ds_time.temperature,
        lon_range=(0, 60),  # Average over this longitude range
        lat_bins=19,
        cmap='plasma',
        title="Temperature Time-Latitude Diagram",
        robust=True
    )
    fig_time_lat.savefig("time_latitude_diagram.png")
    print("Saved time-latitude diagram as 'time_latitude_diagram.png'")
except Exception as e:
    print(f"Error creating time-latitude diagram: {e}")

# Example 3: Time-longitude diagram
print("\nExample 3: Time-longitude diagram")
try:
    fig_time_lon = plotting.plot_time_longitude(
        ds_time.temperature,
        lat_range=(-10, 10),  # Average over this latitude range
        lon_bins=37,
        cmap='cividis',
        title="Temperature Time-Longitude Diagram",
        robust=True
    )
    fig_time_lon.savefig("time_longitude_diagram.png")
    print("Saved time-longitude diagram as 'time_longitude_diagram.png'")
except Exception as e:
    print(f"Error creating time-longitude diagram: {e}")

print("\nAll examples completed!")

print("\nNote: These plotting functions are inspired by and adapted from the EasyGEMS HEALPix")
print("visualization examples available at:")
print("- https://easy.gems.dkrz.de/Processing/healpix/healpix_cartopy.html")
print("- https://easy.gems.dkrz.de/Processing/healpix/time-space.html")
