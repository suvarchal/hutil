#!/usr/bin/env python
# coding: utf-8
"""
Xarray accessor for HEALPix datasets.

This module provides an xarray accessor for HEALPix datasets that allows
for easy selection of data based on latitude, longitude, regions, and more.
"""

__author__ = "Suvarchal K. Cheedela"
__email__ = "suvarchal@duck.com"

import xarray as xr
from . import selection


@xr.register_dataset_accessor("hutil")
class HutilDatasetAccessor:
    """
    Accessor for HEALPix datasets providing selection methods.
    
    This accessor provides methods for selecting data from HEALPix datasets
    based on latitude, longitude, regions, and more.
    """
    
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._has_latlon = False
        
    def _ensure_latlon(self):
        """
        Ensure that the dataset has latitude and longitude coordinates.
        """
        if not self._has_latlon:
            if 'lat' not in self._obj.coords or 'lon' not in self._obj.coords:
                self._obj = selection.add_latlon_coords(self._obj)
            self._has_latlon = True
        return self._obj
    
    def get_info(self):
        """
        Get HEALPix information from the dataset.
        
        Returns:
        --------
        tuple
            (nside, nest, npix)
        """
        return selection.get_healpix_info(self._obj)
    
    def add_latlon_coords(self):
        """
        Add latitude and longitude coordinates to the dataset if they don't exist.
        
        Returns:
        --------
        xarray.Dataset
            Dataset with latitude and longitude coordinates
        """
        self._obj = selection.add_latlon_coords(self._obj)
        self._has_latlon = True
        return self._obj
    
    def select_at_latitude(self, latitude, tolerance=1.0):
        """
        Select data points along a specific latitude with a given tolerance.
        
        Parameters:
        -----------
        latitude : float
            Target latitude in degrees
        tolerance : float, optional
            Tolerance in degrees for latitude selection, default is 1.0
            
        Returns:
        --------
        xarray.Dataset
            Dataset containing only points at the specified latitude (within tolerance)
        """
        self._ensure_latlon()
        return selection.select_at_latitude(self._obj, latitude, tolerance)
    
    def select_at_longitude(self, longitude, tolerance=1.0):
        """
        Select data points along a specific longitude with a given tolerance.
        
        Parameters:
        -----------
        longitude : float
            Target longitude in degrees (-180 to 180)
        tolerance : float, optional
            Tolerance in degrees for longitude selection, default is 1.0
            
        Returns:
        --------
        xarray.Dataset
            Dataset containing only points at the specified longitude (within tolerance)
        """
        self._ensure_latlon()
        return selection.select_at_longitude(self._obj, longitude, tolerance)
    
    def select_at_points(self, points):
        """
        Select data at specific latitude/longitude points using nearest neighbor.
        
        Parameters:
        -----------
        points : list of tuples or array-like
            List of (latitude, longitude) pairs
            
        Returns:
        --------
        xarray.Dataset
            Dataset containing only the nearest points to the specified coordinates
        """
        self._ensure_latlon()
        return selection.select_at_points(self._obj, points)
    
    def select_region(self, lat_min=None, lat_max=None, lon_min=None, lon_max=None, polygon=None, shapefile=None):
        """
        Select data within a region defined by a bounding box, polygon, or shapefile.
        
        Parameters:
        -----------
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
        xarray.Dataset
            Dataset containing only points within the specified region
        """
        self._ensure_latlon()
        return selection.select_region(self._obj, lat_min, lat_max, lon_min, lon_max, polygon, shapefile)
    
    def select_within_polygon(self, polygon):
        """
        Select data within a polygon defined by a list of (lat, lon) pairs.
        
        Parameters:
        -----------
        polygon : list or array-like
            List of (lat, lon) pairs defining a polygon boundary
            
        Returns:
        --------
        xarray.Dataset
            Dataset containing only points within the specified polygon
        """
        self._ensure_latlon()
        return selection.select_within_polygon(self._obj, polygon)
    
    def select_within_shapefile(self, shapefile):
        """
        Select data within polygons defined in a shapefile.
        
        Parameters:
        -----------
        shapefile : str
            Path to a shapefile (.shp) containing polygon boundaries
            
        Returns:
        --------
        xarray.Dataset
            Dataset containing only points within the specified shapefile polygons
        """
        self._ensure_latlon()
        return selection.select_within_shapefile(self._obj, shapefile)
    
    def select_transect(self, lats=None, lons=None, times=None, points=None, method='nearest'):
        """
        Select data along a transect defined by latitude, longitude, and time points.
        
        Parameters:
        -----------
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
        xarray.Dataset
            Dataset containing only the points along the specified transect
        """
        self._ensure_latlon()
        return selection.select_transect(self._obj, lats, lons, times, points, method)
    
    def interpolate_to_grid(self, var_name=None, lat_res=1.0, lon_res=1.0, 
                           lat_min=-90, lat_max=90, lon_min=-180, lon_max=180,
                           method='nearest'):
        """
        Interpolate a HEALPix dataset to a regular lat/lon grid.
        
        Parameters:
        -----------
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
        self._ensure_latlon()
        return selection.interpolate_to_grid(self._obj, var_name, lat_res, lon_res, 
                                           lat_min, lat_max, lon_min, lon_max, method)
    
    def get_value_at_latlon(self, lat, lon, method='nearest'):
        """
        Get the value at a specific latitude/longitude point.
        
        Parameters:
        -----------
        lat : float
            Latitude in degrees
        lon : float
            Longitude in degrees
        method : str, optional
            Interpolation method: 'nearest', 'linear', etc.
            
        Returns:
        --------
        xarray.Dataset or float
            Value at the specified point
        """
        self._ensure_latlon()
        return selection.get_value_at_latlon(self._obj, lat, lon, method)
    
    def plot(self, title=None, cmap='inferno', projection='PlateCarree'):
        """
        Plot HEALPix data with proper coordinate handling.
        
        Parameters:
        -----------
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
        self._ensure_latlon()
        return selection.plot_healpix_selection(self._obj, title, cmap, projection)
    
    def select(self, lat=None, lon=None, time=None, method='nearest'):
        """
        Smart selection function that chooses the appropriate selection method based on input.
        
        Parameters:
        -----------
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
        xarray.Dataset
            Selected data based on the provided parameters
        """
        self._ensure_latlon()
        return selection.select(self._obj, lat, lon, time, method)


@xr.register_dataarray_accessor("hutil")
class HutilDataArrayAccessor:
    """
    Accessor for HEALPix DataArrays providing selection methods.
    
    This accessor provides methods for selecting data from HEALPix DataArrays
    based on latitude, longitude, regions, and more.
    """
    
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._dataset_accessor = HutilDatasetAccessor(xarray_obj.to_dataset())
    
    def get_info(self):
        """
        Get HEALPix information from the DataArray.
        
        Returns:
        --------
        tuple
            (nside, nest, npix)
        """
        return selection.get_healpix_info(self._obj)
    
    def add_latlon_coords(self):
        """
        Add latitude and longitude coordinates to the DataArray if they don't exist.
        
        Returns:
        --------
        xarray.DataArray
            DataArray with latitude and longitude coordinates
        """
        ds = selection.add_latlon_coords(self._obj.to_dataset())
        return ds[self._obj.name]
    
    def select_at_latitude(self, latitude, tolerance=1.0):
        """
        Select data points along a specific latitude with a given tolerance.
        
        Parameters:
        -----------
        latitude : float
            Target latitude in degrees
        tolerance : float, optional
            Tolerance in degrees for latitude selection, default is 1.0
            
        Returns:
        --------
        xarray.DataArray
            DataArray containing only points at the specified latitude (within tolerance)
        """
        return selection.select_at_latitude(self._obj, latitude, tolerance)
    
    def select_at_longitude(self, longitude, tolerance=1.0):
        """
        Select data points along a specific longitude with a given tolerance.
        
        Parameters:
        -----------
        longitude : float
            Target longitude in degrees (-180 to 180)
        tolerance : float, optional
            Tolerance in degrees for longitude selection, default is 1.0
            
        Returns:
        --------
        xarray.DataArray
            DataArray containing only points at the specified longitude (within tolerance)
        """
        return selection.select_at_longitude(self._obj, longitude, tolerance)
    
    def select_at_points(self, points):
        """
        Select data at specific latitude/longitude points using nearest neighbor.
        
        Parameters:
        -----------
        points : list of tuples or array-like
            List of (latitude, longitude) pairs
            
        Returns:
        --------
        xarray.DataArray
            DataArray containing only the nearest points to the specified coordinates
        """
        return selection.select_at_points(self._obj, points)
    
    def select_region(self, lat_min=None, lat_max=None, lon_min=None, lon_max=None, polygon=None, shapefile=None):
        """
        Select data within a region defined by a bounding box, polygon, or shapefile.
        
        Parameters:
        -----------
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
        xarray.DataArray
            DataArray containing only points within the specified region
        """
        return selection.select_region(self._obj, lat_min, lat_max, lon_min, lon_max, polygon, shapefile)
    
    def select_within_polygon(self, polygon):
        """
        Select data within a polygon defined by a list of (lat, lon) pairs.
        
        Parameters:
        -----------
        polygon : list or array-like
            List of (lat, lon) pairs defining a polygon boundary
            
        Returns:
        --------
        xarray.DataArray
            DataArray containing only points within the specified polygon
        """
        return selection.select_within_polygon(self._obj, polygon)
    
    def select_within_shapefile(self, shapefile):
        """
        Select data within polygons defined in a shapefile.
        
        Parameters:
        -----------
        shapefile : str
            Path to a shapefile (.shp) containing polygon boundaries
            
        Returns:
        --------
        xarray.DataArray
            DataArray containing only points within the specified shapefile polygons
        """
        return selection.select_within_shapefile(self._obj, shapefile)
    
    def select_transect(self, lats=None, lons=None, times=None, points=None, method='nearest'):
        """
        Select data along a transect defined by latitude, longitude, and time points.
        
        Parameters:
        -----------
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
        xarray.DataArray
            DataArray containing only the points along the specified transect
        """
        return selection.select_transect(self._obj, lats, lons, times, points, method)
    
    def interpolate_to_grid(self, lat_res=1.0, lon_res=1.0, 
                           lat_min=-90, lat_max=90, lon_min=-180, lon_max=180,
                           method='nearest'):
        """
        Interpolate a HEALPix DataArray to a regular lat/lon grid.
        
        Parameters:
        -----------
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
        return selection.interpolate_to_grid(self._obj, None, lat_res, lon_res, 
                                           lat_min, lat_max, lon_min, lon_max, method)
    
    def get_value_at_latlon(self, lat, lon, method='nearest'):
        """
        Get the value at a specific latitude/longitude point.
        
        Parameters:
        -----------
        lat : float
            Latitude in degrees
        lon : float
            Longitude in degrees
        method : str, optional
            Interpolation method: 'nearest', 'linear', etc.
            
        Returns:
        --------
        float
            Value at the specified point
        """
        return selection.get_value_at_latlon(self._obj, lat, lon, method)
    
    def plot(self, title=None, cmap='inferno', projection='PlateCarree'):
        """
        Plot HEALPix data with proper coordinate handling.
        
        Parameters:
        -----------
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
        return selection.plot_healpix_selection(self._obj, title, cmap, projection)
    
    def select(self, lat=None, lon=None, time=None, method='nearest'):
        """
        Smart selection function that chooses the appropriate selection method based on input.
        
        Parameters:
        -----------
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
        xarray.DataArray
            Selected data based on the provided parameters
        """
        return selection.select(self._obj, lat, lon, time, method)


def register_hutil_accessor():
    """
    Register the hutil accessor with xarray.
    
    This function is called automatically when the package is imported.
    """
    pass  # Registration is handled by the decorators
