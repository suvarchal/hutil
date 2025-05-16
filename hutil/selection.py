#!/usr/bin/env python
# coding: utf-8
"""
HEALPix selection utilities for xarray datasets.

This module provides functions for selecting data from HEALPix datasets
based on latitude, longitude, regions, and more.
"""

__author__ = "Suvarchal K. Cheedela"
__email__ = "suvarchal@duck.com"

import numpy as np
import xarray as xr
import healpy as hp
import matplotlib.pyplot as plt
from matplotlib import colors
import cartopy.crs as ccrs
import shapely.geometry as sgeom
from shapely.geometry import Point, Polygon, MultiPolygon

try:
    import geopandas as gpd
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False


def get_healpix_info(ds):
    """
    Extract HEALPix parameters from a dataset.
    
    Parameters:
    -----------
    ds : xarray.Dataset or xarray.DataArray
        HEALPix dataset
        
    Returns:
    --------
    tuple
        (nside, nest, npix)
    """
    # Convert DataArray to Dataset if needed
    if isinstance(ds, xr.DataArray):
        ds = ds.to_dataset()
    
    # Get HEALPix parameters from crs attributes
    if 'crs' in ds.variables and hasattr(ds.crs, 'healpix_nside'):
        nside = ds.crs.healpix_nside
        nest = ds.crs.healpix_order == 'nest'
        npix = 12 * nside * nside
    else:
        # Try to calculate nside from the cell count
        npix = ds.dims.get('cell')
        if npix:
            nside = hp.npix2nside(npix)
            # Default to nested ordering as that's common
            nest = True
        else:
            raise ValueError("Could not determine HEALPix parameters from dataset")
    
    return nside, nest, npix


def add_latlon_coords(ds):
    """
    Add latitude and longitude coordinates to a HEALPix dataset if they don't exist.
    
    Parameters:
    -----------
    ds : xarray.Dataset or xarray.DataArray
        HEALPix dataset
        
    Returns:
    --------
    xarray.Dataset
        Dataset with latitude and longitude coordinates
    """
    # Convert DataArray to Dataset if needed
    if isinstance(ds, xr.DataArray):
        ds = ds.to_dataset()
    
    if 'lat' in ds.coords and 'lon' in ds.coords:
        return ds
    
    # Get HEALPix parameters
    nside, nest, npix = get_healpix_info(ds)
    
    # Create pixel indices for all cells in the dataset
    pixel_indices = np.arange(ds.dims['cell'])
    
    # Get latitude and longitude for each pixel
    theta, phi = hp.pix2ang(nside, pixel_indices, nest=nest)
    
    # Convert to lat/lon in degrees
    lat = 90 - np.degrees(theta)  # Latitude: 90° at north pole, -90° at south pole
    lon = np.degrees(phi)         # Longitude: 0° to 360°
    
    # Adjust longitude to be in range [-180, 180]
    lon = np.where(lon > 180, lon - 360, lon)
    
    # Add coordinates to the dataset
    ds = ds.assign_coords(lat=('cell', lat), lon=('cell', lon))
    
    return ds


def select_at_latitude(ds, latitude, tolerance=1.0):
    """
    Select data points along a specific latitude with a given tolerance.
    
    Parameters:
    -----------
    ds : xarray.Dataset or xarray.DataArray
        HEALPix dataset with latitude and longitude coordinates
    latitude : float
        Target latitude in degrees
    tolerance : float, optional
        Tolerance in degrees for latitude selection, default is 1.0
        
    Returns:
    --------
    xarray.Dataset or xarray.DataArray
        Dataset containing only points at the specified latitude (within tolerance)
    """
    # Store original type
    is_dataarray = isinstance(ds, xr.DataArray)
    if is_dataarray:
        original_name = ds.name
        ds = ds.to_dataset()
    
    # Ensure lat/lon coordinates exist
    ds = add_latlon_coords(ds)
    
    # Select points within the latitude band
    mask = (ds.lat >= latitude - tolerance) & (ds.lat <= latitude + tolerance)
    result = ds.where(mask, drop=True)
    
    # Return as original type
    if is_dataarray:
        return result[original_name]
    return result


def select_at_longitude(ds, longitude, tolerance=1.0):
    """
    Select data points along a specific longitude with a given tolerance.
    
    Parameters:
    -----------
    ds : xarray.Dataset or xarray.DataArray
        HEALPix dataset with latitude and longitude coordinates
    longitude : float
        Target longitude in degrees (-180 to 180)
    tolerance : float, optional
        Tolerance in degrees for longitude selection, default is 1.0
        
    Returns:
    --------
    xarray.Dataset or xarray.DataArray
        Dataset containing only points at the specified longitude (within tolerance)
    """
    # Store original type
    is_dataarray = isinstance(ds, xr.DataArray)
    if is_dataarray:
        original_name = ds.name
        ds = ds.to_dataset()
    
    # Ensure lat/lon coordinates exist
    ds = add_latlon_coords(ds)
    
    # Normalize longitude to -180 to 180 range
    longitude = ((longitude + 180) % 360) - 180
    
    # Handle longitude wrapping (e.g., -180 and 180 are the same)
    # Calculate the minimum difference considering the wrap-around
    lon_diff = np.minimum(
        np.abs(ds.lon - longitude),
        np.minimum(
            np.abs(ds.lon - (longitude + 360)),
            np.abs(ds.lon - (longitude - 360))
        )
    )
    
    # Select points within the longitude band
    mask = lon_diff <= tolerance
    result = ds.where(mask, drop=True)
    
    # Return as original type
    if is_dataarray:
        return result[original_name]
    return result


def select_at_points(ds, points):
    """
    Select data at specific latitude/longitude points using nearest neighbor.
    
    Parameters:
    -----------
    ds : xarray.Dataset or xarray.DataArray
        HEALPix dataset with latitude and longitude coordinates
    points : list of tuples or array-like
        List of (latitude, longitude) pairs
        
    Returns:
    --------
    xarray.Dataset or xarray.DataArray
        Dataset containing only the nearest points to the specified coordinates
    """
    # Store original type
    is_dataarray = isinstance(ds, xr.DataArray)
    if is_dataarray:
        original_name = ds.name
        ds = ds.to_dataset()
    
    # Get HEALPix parameters
    nside, nest, _ = get_healpix_info(ds)
    
    # Convert lat/lon points to pixel indices
    points = np.array(points)
    
    # Normalize longitudes to be in range [0, 360)
    lons = points[:, 1] % 360
    
    # Convert from lat/lon to theta/phi (healpy uses co-latitude)
    theta = np.radians(90 - points[:, 0])  # Convert latitude to co-latitude in radians
    phi = np.radians(lons)                 # Convert longitude to radians
    
    # Get the pixel indices
    pixels = hp.ang2pix(nside, theta, phi, nest=nest)
    
    # Select the data at those pixels
    result = ds.isel(cell=pixels)
    
    # Return as original type
    if is_dataarray:
        return result[original_name]
    return result


def select_region(ds, lat_min=None, lat_max=None, lon_min=None, lon_max=None, polygon=None, shapefile=None):
    """
    Select data within a region defined by a bounding box, polygon, or shapefile.
    
    Parameters:
    -----------
    ds : xarray.Dataset or xarray.DataArray
        HEALPix dataset with latitude and longitude coordinates
    lat_min, lat_max : float, optional
        Minimum and maximum latitude in degrees for rectangular selection
    lon_min, lon_max : float, optional
        Minimum and maximum longitude in degrees (-180 to 180) for rectangular selection
    polygon : list or array-like, optional
        List of (lat, lon) pairs defining a polygon boundary
    shapefile : str, optional
        Path to a shapefile (.shp) containing polygon boundaries
        
    Returns:
    --------
    xarray.Dataset or xarray.DataArray
        Dataset containing only points within the specified region
        
    Notes:
    ------
    At least one selection method must be provided:
    - Rectangular bounding box (lat_min, lat_max, lon_min, lon_max)
    - Polygon boundary (polygon)
    - Shapefile (shapefile)
    
    If multiple selection methods are provided, they will be combined with OR logic.
    """
    # Store original type
    is_dataarray = isinstance(ds, xr.DataArray)
    if is_dataarray:
        original_name = ds.name
        ds = ds.to_dataset()
    
    # Ensure lat/lon coordinates exist
    ds = add_latlon_coords(ds)
    
    # Initialize an xarray DataArray for the mask with the same coordinates as ds.cell
    mask = xr.DataArray(np.zeros(ds.dims['cell'], dtype=bool), dims=['cell'], coords={'cell': ds.cell})
    
    # Process rectangular bounding box if provided
    if all(x is not None for x in [lat_min, lat_max, lon_min, lon_max]):
        # Normalize longitude range
        lon_min = ((lon_min + 180) % 360) - 180
        lon_max = ((lon_max + 180) % 360) - 180
        
        # Create mask for the region
        lat_mask = (ds.lat >= lat_min) & (ds.lat <= lat_max)
        
        # Handle longitude wrapping
        if lon_min <= lon_max:
            lon_mask = (ds.lon >= lon_min) & (ds.lon <= lon_max)
        else:
            # Crosses the dateline
            lon_mask = (ds.lon >= lon_min) | (ds.lon <= lon_max)
        
        # Combine masks
        rect_mask = lat_mask & lon_mask
        mask = mask | rect_mask
    
    # Process polygon if provided
    if polygon is not None:
        poly_mask_array = _points_in_polygon(ds.lat.values, ds.lon.values, polygon)
        poly_mask = xr.DataArray(poly_mask_array, dims=['cell'], coords={'cell': ds.cell})
        mask = mask | poly_mask
    
    # Process shapefile if provided
    if shapefile is not None:
        if not GEOPANDAS_AVAILABLE:
            raise ImportError("geopandas is required for shapefile support. Install with 'pip install geopandas'.")
        
        shape_mask_array = _points_in_shapefile(ds.lat.values, ds.lon.values, shapefile)
        shape_mask = xr.DataArray(shape_mask_array, dims=['cell'], coords={'cell': ds.cell})
        mask = mask | shape_mask
    
    # Check if any selection method was provided
    if not mask.values.any():
        if all(x is None for x in [lat_min, lat_max, lon_min, lon_max, polygon, shapefile]):
            raise ValueError("At least one selection method must be provided: "
                           "bounding box, polygon, or shapefile.")
        else:
            # No points were selected
            print("Warning: No points were found within the specified region.")
    
    # Apply the mask
    result = ds.where(mask, drop=True)
    
    # Return as original type
    if is_dataarray:
        return result[original_name]
    return result


def _points_in_polygon(lats, lons, polygon):
    """
    Check which points are inside a polygon.
    
    Parameters:
    -----------
    lats : array-like
        Latitude values
    lons : array-like
        Longitude values
    polygon : list or array-like
        List of (lat, lon) pairs defining a polygon boundary
        
    Returns:
    --------
    numpy.ndarray
        Boolean mask where True indicates points inside the polygon
    """
    # Convert polygon to shapely Polygon
    if isinstance(polygon, (Polygon, MultiPolygon)):
        poly = polygon
    else:
        # Convert list of (lat, lon) pairs to shapely Polygon
        # Note: shapely expects (lon, lat) order for coordinates
        poly = Polygon([(lon, lat) for lat, lon in polygon])
    
    # Check each point
    mask = np.zeros(len(lats), dtype=bool)
    for i, (lat, lon) in enumerate(zip(lats, lons)):
        point = Point(lon, lat)  # Note: shapely expects (lon, lat) order
        mask[i] = poly.contains(point)
    
    return mask


def _points_in_shapefile(lats, lons, shapefile):
    """
    Check which points are inside polygons defined in a shapefile.
    
    Parameters:
    -----------
    lats : array-like
        Latitude values
    lons : array-like
        Longitude values
    shapefile : str
        Path to a shapefile (.shp) containing polygon boundaries
        
    Returns:
    --------
    numpy.ndarray
        Boolean mask where True indicates points inside any polygon in the shapefile
    """
    # Load shapefile
    gdf = gpd.read_file(shapefile)
    
    # Initialize mask
    mask = np.zeros(len(lats), dtype=bool)
    
    # Check each point against each polygon in the shapefile
    points = [Point(lon, lat) for lon, lat in zip(lons, lats)]
    
    # For efficiency with large datasets, create a GeoDataFrame of points
    points_gdf = gpd.GeoDataFrame(geometry=points)
    points_gdf.crs = gdf.crs  # Set the same CRS as the shapefile
    
    # Use spatial join to find points within polygons
    # This is much faster than checking each point against each polygon
    joined = gpd.sjoin(points_gdf, gdf, predicate='within', how='left')
    
    # Points that joined successfully are inside a polygon
    mask = ~joined.index_right.isna().values
    
    return mask


def select_within_polygon(ds, polygon):
    """
    Select data within a polygon defined by a list of (lat, lon) pairs.
    
    Parameters:
    -----------
    ds : xarray.Dataset or xarray.DataArray
        HEALPix dataset with latitude and longitude coordinates
    polygon : list or array-like
        List of (lat, lon) pairs defining a polygon boundary
        
    Returns:
    --------
    xarray.Dataset or xarray.DataArray
        Dataset containing only points within the specified polygon
    """
    return select_region(ds, polygon=polygon)


def select_within_shapefile(ds, shapefile):
    """
    Select data within polygons defined in a shapefile.
    
    Parameters:
    -----------
    ds : xarray.Dataset or xarray.DataArray
        HEALPix dataset with latitude and longitude coordinates
    shapefile : str
        Path to a shapefile (.shp) containing polygon boundaries
        
    Returns:
    --------
    xarray.Dataset or xarray.DataArray
        Dataset containing only points within the specified shapefile polygons
    """
    return select_region(ds, shapefile=shapefile)


def interpolate_to_grid(ds, var_name=None, lat_res=1.0, lon_res=1.0, 
                        lat_min=-90, lat_max=90, lon_min=-180, lon_max=180,
                        method='nearest'):
    """
    Interpolate a HEALPix dataset to a regular lat/lon grid.
    
    Parameters:
    -----------
    ds : xarray.Dataset or xarray.DataArray
        HEALPix dataset with latitude and longitude coordinates
    var_name : str, optional
        Name of the variable to interpolate (only needed if ds is a Dataset)
    lat_res, lon_res : float, optional
        Resolution of the output grid in degrees
    lat_min, lat_max, lon_min, lon_max : float, optional
        Boundaries of the output grid
    method : str, optional
        Interpolation method: 'nearest', 'linear', etc.
        
    Returns:
    --------
    xarray.DataArray
        Regular grid data array
    """
    # Handle DataArray input
    is_dataarray = isinstance(ds, xr.DataArray)
    if is_dataarray:
        data_array = ds
        var_name = ds.name
        ds = ds.to_dataset()
    elif var_name is None:
        # If ds is a Dataset and var_name is not provided, use the first data variable
        var_name = list(ds.data_vars)[0]
        data_array = ds[var_name]
    else:
        data_array = ds[var_name]
    
    # Ensure lat/lon coordinates exist
    ds = add_latlon_coords(ds)
    
    # Create target grid
    lats = np.arange(lat_min, lat_max + lat_res, lat_res)
    lons = np.arange(lon_min, lon_max + lon_res, lon_res)
    
    # Create output grid
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')
    output = np.zeros(lat_grid.shape)
    
    # Get HEALPix parameters
    nside, nest, _ = get_healpix_info(ds)
    
    # For each point in the output grid, find the nearest HEALPix pixel
    for i in range(len(lats)):
        for j in range(len(lons)):
            lat = lats[i]
            lon = lons[j]
            
            # Convert lat/lon to HEALPix pixel index
            theta = np.radians(90 - lat)  # Convert latitude to co-latitude
            phi = np.radians(lon % 360)   # Convert longitude to radians
            pixel = hp.ang2pix(nside, theta, phi, nest=nest)
            
            # Get the value at that pixel
            output[i, j] = data_array.isel(cell=pixel).values.item()
    
    # Create a DataArray with the regular grid
    result = xr.DataArray(
        data=output,
        dims=['lat', 'lon'],
        coords={'lat': lats, 'lon': lons},
        name=var_name
    )
    
    return result


def plot_healpix_selection(data, title=None, cmap='inferno', projection='PlateCarree'):
    """
    Plot HEALPix data with proper coordinate handling.
    
    Parameters:
    -----------
    data : xarray.Dataset or xarray.DataArray
        HEALPix dataset or data array with lat/lon coordinates
    title : str, optional
        Plot title
    cmap : str, optional
        Colormap name
    projection : str, optional
        Map projection name from cartopy.crs
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import numpy as np
    
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
        raise ValueError("Data must have lat/lon coordinates. Use add_latlon_coords first.")
    
    # Create figure with map projection
    proj = getattr(ccrs, projection)()
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': proj})
    
    # Get the data values and handle NaN values
    values = data_array.values
    
    # Make sure we have valid data to plot
    if np.all(np.isnan(values)):
        ax.text(0.5, 0.5, 'No valid data to plot', transform=ax.transAxes, 
                ha='center', va='center', fontsize=12)
        ax.coastlines()
        if title:
            plt.title(title)
        return fig
    
    # Get min/max for better color scaling
    vmin = np.nanmin(values)
    vmax = np.nanmax(values)
    
    # Plot the data
    scatter = ax.scatter(
        data_array.lon.values, data_array.lat.values, 
        c=values, 
        cmap=cmap,
        transform=ccrs.PlateCarree(),
        s=20,  # Increased point size for better visibility
        alpha=0.8,  # Slightly more opaque
        vmin=vmin,
        vmax=vmax,
        edgecolor='none'  # Remove point borders for cleaner look
    )
    
    # Add coastlines and grid
    ax.coastlines()
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
    
    # Improve gridline labels
    if hasattr(gl, 'top_labels'):
        gl.top_labels = False
    if hasattr(gl, 'right_labels'):
        gl.right_labels = False
    
    # Add colorbar
    label = data_array.name if data_array.name else 'Value'
    if hasattr(data_array, 'units'):
        label += f' ({data_array.units})'
    elif 'units' in data_array.attrs:
        label += f' ({data_array.attrs["units"]})'
    
    cbar = plt.colorbar(scatter, ax=ax, label=label, pad=0.02, shrink=0.8)
    
    # Set title
    if title:
        plt.title(title)
    
    # Set extent to data bounds with some padding
    lon_min, lon_max = np.nanmin(data_array.lon.values), np.nanmax(data_array.lon.values)
    lat_min, lat_max = np.nanmin(data_array.lat.values), np.nanmax(data_array.lat.values)
    
    # Add padding (10% of range)
    lon_pad = (lon_max - lon_min) * 0.1
    lat_pad = (lat_max - lat_min) * 0.1
    
    # Set map extent if we have a reasonable range
    if lon_max - lon_min > 1 and lat_max - lat_min > 1:
        ax.set_extent([lon_min - lon_pad, lon_max + lon_pad, 
                      lat_min - lat_pad, lat_max + lat_pad], 
                     crs=ccrs.PlateCarree())
    
    return fig


def get_value_at_latlon(ds, lat, lon, method='nearest'):
    """
    Get the value at a specific latitude/longitude point.
    
    Parameters:
    -----------
    ds : xarray.Dataset or xarray.DataArray
        HEALPix dataset with latitude and longitude coordinates
    lat : float
        Latitude in degrees
    lon : float
        Longitude in degrees
    method : str, optional
        Interpolation method: 'nearest', 'linear', etc.
        
    Returns:
    --------
    xarray.Dataset or xarray.DataArray or float
        Value at the specified point
    """
    # Handle DataArray input
    is_dataarray = isinstance(ds, xr.DataArray)
    original_name = None
    
    if is_dataarray:
        original_name = ds.name
        ds_with_coords = add_latlon_coords(ds.to_dataset())
    else:
        ds_with_coords = add_latlon_coords(ds)
    
    # For nearest neighbor method
    if method == 'nearest':
        # Get HEALPix parameters
        nside, nest, _ = get_healpix_info(ds_with_coords)
        
        # Convert lat/lon to pixel index
        theta = np.radians(90 - lat)  # Convert latitude to co-latitude in radians
        phi = np.radians(lon % 360)   # Convert longitude to radians
        
        # Get the pixel index
        pixel = hp.ang2pix(nside, theta, phi, nest=nest)
        
        # Return the value at that pixel
        if is_dataarray:
            return ds_with_coords[original_name].isel(cell=pixel).values.item()
        else:
            return ds_with_coords.isel(cell=pixel)
    
    # For other interpolation methods
    else:
        # Create a single-point dataset for interpolation
        point = xr.Dataset(coords={'lat': [lat], 'lon': [lon]})
        
        # Interpolate
        if is_dataarray:
            result = ds_with_coords[original_name].interp(lat=point.lat, lon=point.lon, method=method)
            return result.values.item()
        else:
            return ds_with_coords.interp(lat=point.lat, lon=point.lon, method=method)


def select_transect(ds, lats=None, lons=None, times=None, points=None, method='nearest'):
    """
    Select data along a transect defined by latitude, longitude, and time points.
    
    Parameters:
    -----------
    ds : xarray.Dataset or xarray.DataArray
        HEALPix dataset with latitude and longitude coordinates
    lats : array-like, optional
        Array of latitude values for the transect points
    lons : array-like, optional
        Array of longitude values for the transect points
    times : array-like, optional
        Array of time values for the transect points
    points : list of tuples, optional
        List of (lat, lon, time) tuples defining the transect
    method : str, optional
        Interpolation method: 'nearest', 'linear', etc.
        
    Returns:
    --------
    xarray.Dataset or xarray.DataArray
        Dataset containing only the points along the specified transect
        
    Notes:
    ------
    Either provide:
    - Individual arrays for lats, lons, and times (all must be the same length)
    - A list of (lat, lon, time) tuples in the points parameter
    """
    # Check input parameters
    if points is not None:
        # Convert list of tuples to separate arrays
        points = np.array(points)
        lats = points[:, 0]
        lons = points[:, 1]
        times = points[:, 2]
    elif lats is not None and lons is not None and times is not None:
        # Ensure all arrays have the same length
        if len(lats) != len(lons) or len(lats) != len(times):
            raise ValueError("lats, lons, and times must all have the same length")
    else:
        raise ValueError("Either provide points as a list of (lat, lon, time) tuples or separate lats, lons, and times arrays")
    
    # Store original type
    is_dataarray = isinstance(ds, xr.DataArray)
    if is_dataarray:
        original_name = ds.name
        ds = ds.to_dataset()
    
    # Ensure lat/lon coordinates exist
    ds = add_latlon_coords(ds)
    
    # Create an empty list to store results for each time point
    results = []
    
    # Process each point in the transect
    for lat, lon, time in zip(lats, lons, times):
        # Select the data at the specific time
        if 'time' in ds.dims or 'time' in ds.coords:
            ds_time = ds.sel(time=time, method=method)
        else:
            ds_time = ds
        
        # Get the value at the specific lat/lon point
        if method == 'nearest':
            # Get HEALPix parameters
            nside, nest, _ = get_healpix_info(ds_time)
            
            # Convert lat/lon to pixel index
            theta = np.radians(90 - lat)  # Convert latitude to co-latitude in radians
            phi = np.radians(lon % 360)   # Convert longitude to radians
            
            # Get the pixel index
            pixel = hp.ang2pix(nside, theta, phi, nest=nest)
            
            # Get the data at that pixel and add to results
            point_data = ds_time.isel(cell=pixel)
            
            # Add lat/lon/time coordinates if they don't exist
            if 'lat' not in point_data.coords:
                point_data = point_data.assign_coords(lat=lat)
            if 'lon' not in point_data.coords:
                point_data = point_data.assign_coords(lon=lon)
            if 'time' not in point_data.coords and 'time' in ds.coords:
                point_data = point_data.assign_coords(time=time)
                
            results.append(point_data)
        else:
            # For other interpolation methods
            # Create a single-point dataset for interpolation
            point = xr.Dataset(coords={'lat': [lat], 'lon': [lon]})
            
            # Interpolate and add to results
            point_data = ds_time.interp(lat=point.lat, lon=point.lon, method=method)
            
            # Add time coordinate if it doesn't exist
            if 'time' not in point_data.coords and 'time' in ds.coords:
                point_data = point_data.assign_coords(time=time)
                
            results.append(point_data)
    
    # Combine all points into a single dataset
    if results:
        # Create a new dimension for the transect points
        result = xr.concat(results, dim='point')
        
        # Add point index as a coordinate
        result = result.assign_coords(point=np.arange(len(results)))
        
        # Return as original type
        if is_dataarray:
            return result[original_name]
        return result
    else:
        # No points were found
        if is_dataarray:
            return xr.DataArray()
        return xr.Dataset()


def select(ds, lat=None, lon=None, time=None, method='nearest'):
    """
    Smart selection function that chooses the appropriate selection method based on input.
    
    Parameters:
    -----------
    ds : xarray.Dataset or xarray.DataArray
        HEALPix dataset with latitude and longitude coordinates
    lat : float, slice, or array-like, optional
        Latitude value(s) for selection
    lon : float, slice, or array-like, optional
        Longitude value(s) for selection
    time : datetime, str, or array-like, optional
        Time value(s) for selection
    method : str, optional
        Interpolation method: 'nearest', 'linear', etc.
        
    Returns:
    --------
    xarray.Dataset or xarray.DataArray
        Selected data based on the provided parameters
        
    Notes:
    ------
    This function intelligently selects the appropriate selection method based on the input:
    - If lat is a single value and lon is None: select_at_latitude
    - If lon is a single value and lat is None: select_at_longitude
    - If lat and lon are both single values: get_value_at_latlon
    - If lat and lon are both slices: select_region (rectangular)
    - If lat, lon, and time are arrays of the same length: select_transect
    """
    # Case 1: Select at a specific latitude
    if isinstance(lat, (int, float)) and lon is None:
        return select_at_latitude(ds, latitude=lat)
    
    # Case 2: Select at a specific longitude
    elif isinstance(lon, (int, float)) and lat is None:
        return select_at_longitude(ds, longitude=lon)
    
    # Case 3: Select at a specific lat/lon point
    elif isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
        if time is not None:
            # If time is provided, select at the specific time first
            if 'time' in ds.dims or 'time' in ds.coords:
                ds = ds.sel(time=time, method=method)
        return get_value_at_latlon(ds, lat=lat, lon=lon, method=method)
    
    # Case 4: Select within a rectangular region
    elif isinstance(lat, slice) and isinstance(lon, slice):
        lat_min = lat.start if lat.start is not None else -90
        lat_max = lat.stop if lat.stop is not None else 90
        lon_min = lon.start if lon.start is not None else -180
        lon_max = lon.stop if lon.stop is not None else 180
        
        # If time is provided, select at the specific time first
        if time is not None and ('time' in ds.dims or 'time' in ds.coords):
            ds = ds.sel(time=time, method=method)
            
        return select_region(ds, lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max)
    
    # Case 5: Select a transect (arrays of lat, lon, and optionally time)
    elif hasattr(lat, '__len__') and hasattr(lon, '__len__') and not isinstance(lat, (str, slice)) and not isinstance(lon, (str, slice)):
        if time is None:
            # If time is not provided, use the first time in the dataset or create a dummy time array
            if 'time' in ds.dims or 'time' in ds.coords:
                times = np.array([ds.time.values[0]] * len(lat))
            else:
                times = np.array([np.datetime64('2000-01-01')] * len(lat))
        elif not hasattr(time, '__len__') or isinstance(time, (str, np.datetime64)):
            # If time is a single value, repeat it for each lat/lon point
            times = np.array([time] * len(lat))
        else:
            # Use the provided time array
            times = time
            
        return select_transect(ds, lats=lat, lons=lon, times=times, method=method)
    
    # Case 6: Select at specific points (list of lat/lon pairs)
    elif isinstance(lat, list) and all(isinstance(item, tuple) and len(item) == 2 for item in lat) and lon is None:
        return select_at_points(ds, points=lat)
    
    else:
        raise ValueError("Invalid selection parameters. Please provide valid lat/lon/time values.")

