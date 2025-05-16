#!/usr/bin/env python
# coding: utf-8
"""
Enhanced plotting utilities for HEALPix datasets.

This module provides advanced plotting functions for HEALPix datasets,
including map plots with cartopy and time-space cross-section plots.

Acknowledgments:
These plotting functions are inspired by and adapted from the EasyGEMS HEALPix
visualization examples available at:
- https://easy.gems.dkrz.de/Processing/healpix/healpix_cartopy.html
- https://easy.gems.dkrz.de/Processing/healpix/time-space.html
"""

__author__ = "Suvarchal K. Cheedela"
__email__ = "suvarchal@duck.com"

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point

from .selection import add_latlon_coords


def plot_map(data, projection='PlateCarree', cmap='inferno', title=None, 
             vmin=None, vmax=None, robust=True, figsize=(10, 6),
             coastlines=True, add_colorbar=True, cbar_label=None,
             central_longitude=0, extent=None):
    """
    Create an enhanced map plot of HEALPix data using cartopy.
    
    Parameters:
    -----------
    data : xarray.Dataset or xarray.DataArray
        HEALPix dataset or data array with lat/lon coordinates
    projection : str, optional
        Map projection name from cartopy.crs (default: 'PlateCarree')
    cmap : str, optional
        Colormap name (default: 'inferno')
    title : str, optional
        Plot title
    vmin, vmax : float, optional
        Min and max values for colormap scaling
    robust : bool, optional
        If True and vmin/vmax not provided, use robust quantile-based scaling
    figsize : tuple, optional
        Figure size (width, height) in inches
    coastlines : bool, optional
        Whether to add coastlines to the map
    add_colorbar : bool, optional
        Whether to add a colorbar
    cbar_label : str, optional
        Label for the colorbar
    central_longitude : float, optional
        Central longitude for the projection
    extent : list, optional
        Map extent [lon_min, lon_max, lat_min, lat_max]
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    # Ensure we have a DataArray
    if isinstance(data, xr.Dataset):
        if len(data.data_vars) == 0:
            raise ValueError("Dataset has no data variables")
        # Use the first data variable
        var_name = list(data.data_vars)[0]
        data_array = data[var_name]
    else:
        data_array = data
    
    # Ensure lat/lon coordinates exist
    if 'lat' not in data_array.coords or 'lon' not in data_array.coords:
        data_array = add_latlon_coords(data_array)
    
    # Create figure with map projection
    try:
        proj_class = getattr(ccrs, projection)
        if projection in ['Robinson', 'Mollweide', 'PlateCarree', 'Orthographic']:
            proj = proj_class(central_longitude=central_longitude)
        else:
            proj = proj_class()
        
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': proj})
    except Exception as e:
        # Fallback to PlateCarree if the projection fails
        print(f"Warning: Projection {projection} failed, falling back to PlateCarree. Error: {e}")
        proj = ccrs.PlateCarree()
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': proj})
    
    # Get the data values and handle NaN values
    values = data_array.values
    lons = data_array.lon.values
    lats = data_array.lat.values
    
    # Make sure we have valid data to plot
    if np.all(np.isnan(values)):
        ax.text(0.5, 0.5, 'No valid data to plot', transform=ax.transAxes, 
                ha='center', va='center', fontsize=12)
        if coastlines:
            ax.coastlines()
        if title:
            plt.title(title)
        return fig
    
    # Get min/max for better color scaling if not provided
    if vmin is None or vmax is None:
        if robust:
            if vmin is None:
                vmin = np.nanquantile(values, 0.02)
            if vmax is None:
                vmax = np.nanquantile(values, 0.98)
        else:
            if vmin is None:
                vmin = np.nanmin(values)
            if vmax is None:
                vmax = np.nanmax(values)
    
    # Filter out NaN values to avoid plotting issues
    valid_mask = ~np.isnan(values) & ~np.isnan(lons) & ~np.isnan(lats)
    if np.sum(valid_mask) == 0:
        ax.text(0.5, 0.5, 'No valid data to plot', transform=ax.transAxes, 
                ha='center', va='center', fontsize=12)
        if coastlines:
            ax.coastlines()
        if title:
            plt.title(title)
        return fig
        
    plot_lons = lons[valid_mask]
    plot_lats = lats[valid_mask]
    plot_values = values[valid_mask]
    
    # Plot the data
    scatter = ax.scatter(
        plot_lons, plot_lats, 
        c=plot_values, 
        cmap=cmap,
        transform=ccrs.PlateCarree(),
        s=20,  # Increased point size for better visibility
        alpha=0.8,  # Slightly more opaque
        vmin=vmin,
        vmax=vmax,
        edgecolor='none'  # Remove point borders for cleaner look
    )
    
    # Add coastlines and grid
    if coastlines:
        ax.coastlines(linewidth=0.5)
    
    # Add gridlines with proper handling for different projections
    try:
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
        
        # Improve gridline labels
        if hasattr(gl, 'top_labels'):
            gl.top_labels = False
        if hasattr(gl, 'right_labels'):
            gl.right_labels = False
    except Exception as e:
        # Some projections may not support all gridline features
        print(f"Warning: Could not add all gridline features. Error: {e}")
        try:
            gl = ax.gridlines(linewidth=0.5, alpha=0.5)
        except:
            # If even simple gridlines fail, continue without them
            pass
    
    # Add colorbar
    if add_colorbar:
        if cbar_label is None:
            cbar_label = data_array.name if data_array.name else 'Value'
            if hasattr(data_array, 'units'):
                cbar_label += f' ({data_array.units})'
            elif 'units' in data_array.attrs:
                cbar_label += f' ({data_array.attrs["units"]})'
        
        cbar = plt.colorbar(scatter, ax=ax, label=cbar_label, pad=0.02, shrink=0.8)
    
    # Set title
    if title:
        plt.title(title)
    
    # Set extent if provided
    if extent is not None:
        try:
            ax.set_extent(extent, crs=ccrs.PlateCarree())
        except Exception as e:
            print(f"Warning: Could not set custom extent. Error: {e}")
    else:
        # Try to set a reasonable extent based on data
        try:
            # Set extent to data bounds with some padding
            if len(plot_lons) > 0 and len(plot_lats) > 0:
                lon_min, lon_max = np.min(plot_lons), np.max(plot_lons)
                lat_min, lat_max = np.min(plot_lats), np.max(plot_lats)
                
                # Add padding (10% of range)
                lon_range = lon_max - lon_min
                lat_range = lat_max - lat_min
                
                if lon_range > 0 and lat_range > 0:
                    lon_pad = max(1, lon_range * 0.1)
                    lat_pad = max(1, lat_range * 0.1)
                    
                    # For global projections, don't set extent
                    if projection not in ['Robinson', 'Mollweide']:
                        ax.set_extent([lon_min - lon_pad, lon_max + lon_pad, 
                                      lat_min - lat_pad, lat_max + lat_pad], 
                                     crs=ccrs.PlateCarree())
        except Exception as e:
            print(f"Warning: Could not set automatic extent. Error: {e}")
            # Continue without setting extent
    
    return fig


def plot_time_latitude(data, lon_range=None, lat_bins=19, time_dim='time',
                      cmap='inferno', title=None, vmin=None, vmax=None, 
                      robust=True, figsize=(10, 6), add_colorbar=True, 
                      cbar_label=None):
    """
    Create a time-latitude cross-section plot (Hovmöller diagram).
    
    Parameters:
    -----------
    data : xarray.Dataset or xarray.DataArray
        HEALPix dataset or data array with lat/lon coordinates and time dimension
    lon_range : tuple, optional
        Longitude range (min, max) to average over
    lat_bins : int, optional
        Number of latitude bins
    time_dim : str, optional
        Name of the time dimension
    cmap : str, optional
        Colormap name
    title : str, optional
        Plot title
    vmin, vmax : float, optional
        Min and max values for colormap scaling
    robust : bool, optional
        If True and vmin/vmax not provided, use robust quantile-based scaling
    figsize : tuple, optional
        Figure size (width, height) in inches
    add_colorbar : bool, optional
        Whether to add a colorbar
    cbar_label : str, optional
        Label for the colorbar
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    # Ensure we have a DataArray
    if isinstance(data, xr.Dataset):
        if len(data.data_vars) == 0:
            raise ValueError("Dataset has no data variables")
        # Use the first data variable
        var_name = list(data.data_vars)[0]
        data_array = data[var_name]
    else:
        data_array = data
    
    # Ensure lat/lon coordinates exist
    if 'lat' not in data_array.coords or 'lon' not in data_array.coords:
        data_array = add_latlon_coords(data_array)
    
    # Filter by longitude range if specified
    if lon_range is not None:
        lon_min, lon_max = lon_range
        lon_mask = (data_array.lon >= lon_min) & (data_array.lon <= lon_max)
        data_array = data_array.where(lon_mask, drop=True)
    
    # Create latitude bins
    lat_edges = np.linspace(-90, 90, lat_bins + 1)
    lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2
    
    # Create a new array to store binned data
    times = data_array[time_dim].values
    result = np.zeros((len(times), lat_bins))
    result[:] = np.nan
    
    # Bin the data by latitude
    for i, (lat_min, lat_max) in enumerate(zip(lat_edges[:-1], lat_edges[1:])):
        lat_mask = (data_array.lat >= lat_min) & (data_array.lat < lat_max)
        if np.any(lat_mask):
            # Average over all points in this latitude band
            lat_data = data_array.where(lat_mask, drop=True)
            # Average over longitude (all points in this latitude band)
            lat_mean = lat_data.mean(dim='cell')
            result[:, i] = lat_mean.values
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get min/max for better color scaling if not provided
    if vmin is None or vmax is None:
        if robust:
            if vmin is None:
                vmin = np.nanquantile(result, 0.02)
            if vmax is None:
                vmax = np.nanquantile(result, 0.98)
        else:
            if vmin is None:
                vmin = np.nanmin(result)
            if vmax is None:
                vmax = np.nanmax(result)
    
    # Plot the data
    im = ax.pcolormesh(
        lat_centers, 
        times, 
        result, 
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading='auto'
    )
    
    # Format the axes
    ax.set_xlabel('Latitude (°N)')
    ax.set_ylabel(f'{time_dim.capitalize()}')
    
    # Format time axis if using datetime
    if np.issubdtype(times.dtype, np.datetime64):
        ax.yaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
        ax.yaxis.set_minor_locator(mdates.MonthLocator())
        ax.yaxis.set_major_formatter(
            mdates.ConciseDateFormatter(ax.yaxis.get_major_locator())
        )
    
    # Use equal-area scaling for latitude axis
    ax.set_xscale(
        "function",
        functions=(
            lambda d: np.sin(np.deg2rad(d)),
            lambda d: np.rad2deg(np.arcsin(np.clip(d, -1, 1))),
        ),
    )
    
    # Set latitude ticks
    ax.set_xticks([-90, -60, -30, 0, 30, 60, 90])
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Add colorbar
    if add_colorbar:
        if cbar_label is None:
            cbar_label = data_array.name if data_array.name else 'Value'
            if hasattr(data_array, 'units'):
                cbar_label += f' ({data_array.units})'
            elif 'units' in data_array.attrs:
                cbar_label += f' ({data_array.attrs["units"]})'
        
        cbar = plt.colorbar(im, ax=ax, label=cbar_label, pad=0.02, shrink=0.8)
    
    # Set title
    if title:
        plt.title(title)
    else:
        var_name = data_array.name if data_array.name else 'Variable'
        lon_text = f" (Lon: {lon_min:.1f}°-{lon_max:.1f}°)" if lon_range else ""
        plt.title(f"{var_name} Time-Latitude Diagram{lon_text}")
    
    return fig


def plot_time_longitude(data, lat_range=None, lon_bins=37, time_dim='time',
                       cmap='inferno', title=None, vmin=None, vmax=None, 
                       robust=True, figsize=(10, 6), add_colorbar=True, 
                       cbar_label=None):
    """
    Create a time-longitude cross-section plot (Hovmöller diagram).
    
    Parameters:
    -----------
    data : xarray.Dataset or xarray.DataArray
        HEALPix dataset or data array with lat/lon coordinates and time dimension
    lat_range : tuple, optional
        Latitude range (min, max) to average over
    lon_bins : int, optional
        Number of longitude bins
    time_dim : str, optional
        Name of the time dimension
    cmap : str, optional
        Colormap name
    title : str, optional
        Plot title
    vmin, vmax : float, optional
        Min and max values for colormap scaling
    robust : bool, optional
        If True and vmin/vmax not provided, use robust quantile-based scaling
    figsize : tuple, optional
        Figure size (width, height) in inches
    add_colorbar : bool, optional
        Whether to add a colorbar
    cbar_label : str, optional
        Label for the colorbar
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    # Ensure we have a DataArray
    if isinstance(data, xr.Dataset):
        if len(data.data_vars) == 0:
            raise ValueError("Dataset has no data variables")
        # Use the first data variable
        var_name = list(data.data_vars)[0]
        data_array = data[var_name]
    else:
        data_array = data
    
    # Ensure lat/lon coordinates exist
    if 'lat' not in data_array.coords or 'lon' not in data_array.coords:
        data_array = add_latlon_coords(data_array)
    
    # Filter by latitude range if specified
    if lat_range is not None:
        lat_min, lat_max = lat_range
        lat_mask = (data_array.lat >= lat_min) & (data_array.lat <= lat_max)
        data_array = data_array.where(lat_mask, drop=True)
    
    # Create longitude bins
    lon_edges = np.linspace(0, 360, lon_bins + 1)
    lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2
    
    # Create a new array to store binned data
    times = data_array[time_dim].values
    result = np.zeros((len(times), lon_bins))
    result[:] = np.nan
    
    # Bin the data by longitude
    for i, (lon_min, lon_max) in enumerate(zip(lon_edges[:-1], lon_edges[1:])):
        # Handle wrapping around 0/360
        if lon_min < 0:
            lon_mask = ((data_array.lon >= lon_min + 360) | (data_array.lon < lon_max))
        else:
            lon_mask = ((data_array.lon >= lon_min) & (data_array.lon < lon_max))
        
        if np.any(lon_mask):
            # Average over all points in this longitude band
            lon_data = data_array.where(lon_mask, drop=True)
            # Average over latitude (all points in this longitude band)
            lon_mean = lon_data.mean(dim='cell')
            result[:, i] = lon_mean.values
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get min/max for better color scaling if not provided
    if vmin is None or vmax is None:
        if robust:
            if vmin is None:
                vmin = np.nanquantile(result, 0.02)
            if vmax is None:
                vmax = np.nanquantile(result, 0.98)
        else:
            if vmin is None:
                vmin = np.nanmin(result)
            if vmax is None:
                vmax = np.nanmax(result)
    
    # Plot the data
    im = ax.pcolormesh(
        lon_centers, 
        times, 
        result, 
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading='auto'
    )
    
    # Format the axes
    ax.set_xlabel('Longitude (°E)')
    ax.set_ylabel(f'{time_dim.capitalize()}')
    
    # Format time axis if using datetime
    if np.issubdtype(times.dtype, np.datetime64):
        ax.yaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
        ax.yaxis.set_minor_locator(mdates.MonthLocator())
        ax.yaxis.set_major_formatter(
            mdates.ConciseDateFormatter(ax.yaxis.get_major_locator())
        )
    
    # Set longitude ticks
    ax.set_xticks(np.arange(0, 361, 60))
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Add colorbar
    if add_colorbar:
        if cbar_label is None:
            cbar_label = data_array.name if data_array.name else 'Value'
            if hasattr(data_array, 'units'):
                cbar_label += f' ({data_array.units})'
            elif 'units' in data_array.attrs:
                cbar_label += f' ({data_array.attrs["units"]})'
        
        cbar = plt.colorbar(im, ax=ax, label=cbar_label, pad=0.02, shrink=0.8)
    
    # Set title
    if title:
        plt.title(title)
    else:
        var_name = data_array.name if data_array.name else 'Variable'
        lat_text = f" (Lat: {lat_min:.1f}°-{lat_max:.1f}°)" if lat_range else ""
        plt.title(f"{var_name} Time-Longitude Diagram{lat_text}")
    
    return fig
