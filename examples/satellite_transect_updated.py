#!/usr/bin/env python
# coding: utf-8
"""
Creating Satellite Transects with HUtil

This script demonstrates how to create a transect from satellite orbit data using the
`hutil` package. We'll use a real satellite orbit file and show how to extract data
along the satellite track from a HEALPix dataset.

Acknowledgments:
The plotting functions are inspired by and adapted from the EasyGEMS HEALPix
visualization examples available at:
- https://easy.gems.dkrz.de/Processing/healpix/healpix_cartopy.html
- https://easy.gems.dkrz.de/Processing/healpix/time-space.html
"""

import os
import numpy as np
import xarray as xr
import healpy as hp
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import datetime
from pathlib import Path

# Import hutil
import hutil
from hutil import selection
from hutil import plotting

# Set matplotlib to display plots inline if running in a notebook
# %matplotlib inline
# plt.rcParams['figure.figsize'] = [10, 6]  # Set default figure size

# Load the satellite orbit data
data_dir = Path(hutil.__file__).parent / 'data'
orbit_file = data_dir / 'ECA_J_CPR_NOM_1BS_20240617T1356_20240617T1407_00306F_vAb_location.nc'

orbit_data = xr.open_dataset(orbit_file)
print(f'Loaded orbit data from {orbit_file.name}')
print(orbit_data)

# Extract latitude, longitude, and time from the orbit data
lats = orbit_data.latitude.values
lons = orbit_data.longitude.values
times = orbit_data.time.values

print(f'Orbit data contains {len(lats)} points')
print(f'Latitude range: {lats.min():.2f}° to {lats.max():.2f}°')
print(f'Longitude range: {lons.min():.2f}° to {lons.max():.2f}°')
print(f'Time range: {np.datetime_as_string(times[0])} to {np.datetime_as_string(times[-1])}')

# Plot the satellite orbit track using the enhanced plotting functionality
fig = plt.figure(figsize=(12, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines(linewidth=0.5)
ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)

# Plot the orbit track
ax.plot(lons, lats, 'r-', transform=ccrs.PlateCarree(), linewidth=2, label='Satellite Track')
ax.scatter(lons[0], lats[0], color='green', s=100, label='Start', transform=ccrs.PlateCarree())
ax.scatter(lons[-1], lats[-1], color='blue', s=100, label='End', transform=ccrs.PlateCarree())

plt.title('Satellite Orbit Track')
plt.legend()
plt.savefig('satellite_track.png')
plt.close()
print("Saved satellite track plot as 'satellite_track.png'")

# Create a HEALPix dataset
nside = 128  # Higher resolution for better transect
npix = hp.nside2npix(nside)
nest = True

# Get the latitude and longitude for each HEALPix pixel
hp_lats, hp_lons = hp.pix2ang(nside, np.arange(npix), nest=nest, lonlat=True)

# Create a temperature field that varies with latitude and longitude
# Base temperature + latitudinal gradient + longitudinal wave
temperature = 15 + 30 * np.cos(np.radians(hp_lats)) + 5 * np.sin(np.radians(hp_lons))

# Create a dataset with a time dimension (we'll use the satellite time range)
start_time = np.datetime64(times[0])
end_time = np.datetime64(times[-1])
time_range = np.linspace(0, 1, 10)  # 10 time steps
time_values = start_time + (end_time - start_time) * time_range

# Create a time-varying temperature field
temp_time = np.zeros((len(time_values), npix))
for i, t in enumerate(time_range):
    # Add a time-varying component (warming over time)
    temp_time[i, :] = temperature + 2 * t  # Temperature increases by 2°C over the time period

# Create the dataset
ds = xr.Dataset(
    data_vars={
        'temperature': (['time', 'cell'], temp_time)
    },
    coords={
        'time': time_values,
        'cell': np.arange(npix)
    },
    attrs={
        'nside': nside,
        'nest': nest
    }
)

print(f'Created HEALPix dataset with nside={nside}, npix={npix}, and {len(time_values)} time steps')
print(ds)

# Add latitude and longitude coordinates to the dataset
ds_with_latlon = ds.hutil.add_latlon_coords()
print('Added lat/lon coordinates to dataset')
print(ds_with_latlon)

# Select the first time step
ds_t0 = ds_with_latlon.sel(time=ds_with_latlon.time[0])

# Plot the temperature field using the enhanced plotting functionality
fig = plotting.plot_map(
    ds_t0.temperature,
    projection='PlateCarree',
    cmap='inferno',
    title=f'Temperature at {np.datetime_as_string(ds_t0.time.values, unit="m")} (°C)',
    robust=True,
    figsize=(12, 8)
)
plt.savefig('temperature_map.png')
plt.close()
print("Saved temperature map as 'temperature_map.png'")

# For demonstration, let's use a subset of the orbit points to make the example faster
# In a real application, you might want to use all points
step = 10  # Use every 10th point
subset_lats = lats[::step]
subset_lons = lons[::step]
subset_times = times[::step]

print(f'Using {len(subset_lats)} points for the transect')

# Create a transect along the satellite track
transect = selection.select_transect(
    ds_with_latlon,
    lats=subset_lats,
    lons=subset_lons,
    times=subset_times,
    method='nearest'
)

print(f'Created transect with {len(transect.point)} points')
print(transect)

# Plot the transect on a map using the enhanced plotting functionality
fig = plt.figure(figsize=(12, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines(linewidth=0.5)
ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)

# Plot the full orbit track
ax.plot(lons, lats, 'k-', transform=ccrs.PlateCarree(), linewidth=1, alpha=0.5, label='Full Orbit')

# Plot the transect points
scatter = ax.scatter(
    transect.lon.values, 
    transect.lat.values, 
    c=transect.temperature.values, 
    cmap='inferno',
    s=50, 
    transform=ccrs.PlateCarree(),
    label='Transect Points'
)

# Add a colorbar
cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
cbar.set_label('Temperature (°C)')

plt.title('Satellite Transect')
plt.legend()
plt.savefig('satellite_transect.png')
plt.close()
print("Saved satellite transect plot as 'satellite_transect.png'")

# Calculate the distance along the transect
def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r

# Calculate the distance along the transect
transect_lats = transect.lat.values
transect_lons = transect.lon.values

distances = [0]  # Start with 0 km
for i in range(1, len(transect_lats)):
    d = haversine_distance(
        transect_lats[i-1], transect_lons[i-1],
        transect_lats[i], transect_lons[i]
    )
    distances.append(distances[-1] + d)

# Plot the temperature along the transect
plt.figure(figsize=(12, 6))
plt.plot(distances, transect.temperature.values, 'r-', linewidth=2)
scatter = plt.scatter(distances, transect.temperature.values, c=transect.temperature.values, cmap='inferno', s=50)
plt.xlabel('Distance along transect (km)')
plt.ylabel('Temperature (°C)')
plt.title('Temperature along Satellite Transect')
plt.grid(True)
plt.colorbar(scatter, label='Temperature (°C)')
plt.savefig('temperature_along_transect.png')
plt.close()
print("Saved temperature along transect plot as 'temperature_along_transect.png'")

# Create a transect along the satellite track using the accessor
transect_accessor = ds_with_latlon.hutil.select_transect(
    lats=subset_lats,
    lons=subset_lons,
    times=subset_times,
    method='nearest'
)

print(f'Created transect with {len(transect_accessor.point)} points using accessor')
print(transect_accessor)

# Check if the transects are identical
identical = np.allclose(transect.temperature.values, transect_accessor.temperature.values)
print(f'Transects are identical: {identical}')

# Path to the Uganda shapefile
uganda_shapefile = data_dir / 'combined_uganda.shp'

# Select data within Uganda using the direct function call
uganda_data = selection.select_within_shapefile(ds_with_latlon.isel(time=0), uganda_shapefile)

print(f'Selected {len(uganda_data.cell)} points within Uganda')
print(uganda_data)

# Plot the data within Uganda using the enhanced plotting functionality
fig = plotting.plot_map(
    uganda_data.temperature,
    projection='PlateCarree',
    cmap='inferno',
    title='Temperature in Uganda (°C)',
    robust=True,
    figsize=(10, 8)
)
plt.savefig('uganda_temperature.png')
plt.close()
print("Saved Uganda temperature map as 'uganda_temperature.png'")

# Select data within Uganda using the accessor
uganda_data_accessor = ds_with_latlon.isel(time=0).hutil.select_within_shapefile(uganda_shapefile)

print(f'Selected {len(uganda_data_accessor.cell)} points within Uganda using accessor')
print(uganda_data_accessor)

# Plot the data within Uganda using the accessor with enhanced plotting
fig = uganda_data_accessor.hutil.plot(
    title='Temperature in Uganda (°C) - Accessor',
    cmap='inferno',
    projection='PlateCarree',
    robust=True,
    figsize=(10, 8)
)
plt.savefig('uganda_temperature_accessor.png')
plt.close()
print("Saved Uganda temperature map (accessor) as 'uganda_temperature_accessor.png'")

print("\nNote: The plotting functions in this example are inspired by and adapted from the EasyGEMS HEALPix")
print("visualization examples available at:")
print("- https://easy.gems.dkrz.de/Processing/healpix/healpix_cartopy.html")
print("- https://easy.gems.dkrz.de/Processing/healpix/time-space.html")
