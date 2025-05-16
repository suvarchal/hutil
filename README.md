# HUtil: HEALPix Utilities for xarray

[![PyPI version](https://badge.fury.io/py/hutil.svg)](https://badge.fury.io/py/hutil)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

HUtil provides utilities for working with HEALPix datasets in xarray, including selection methods for points, regions, and transects.

## Features

- Select data at specific latitude/longitude points
- Select data along latitude/longitude lines
- Select data within regions (rectangular, polygon, or shapefile)
- Select data along transects (series of lat/lon/time points)
- Interpolate HEALPix data to regular lat/lon grids
- Plot HEALPix data with proper coordinate handling
- Smart selection based on input types

## Installation

```bash
pip install hutil
```

For shapefile support, install with:

```bash
pip install hutil[shapefile]
```

## Usage

HUtil can be used in two ways:

1. As standalone functions:

```python
import xarray as xr
from hutil import select_at_latitude, add_latlon_coords

# Load a HEALPix dataset
ds = xr.open_dataset('healpix_data.nc')

# Add lat/lon coordinates if they don't exist
ds = add_latlon_coords(ds)

# Select data along the equator
equator_data = select_at_latitude(ds, latitude=0, tolerance=1.0)
```

2. As an xarray accessor:

```python
import xarray as xr
import hutil

# Load a HEALPix dataset
ds = xr.open_dataset('healpix_data.nc')

# Select data along the equator
equator_data = ds.hutil.select_at_latitude(latitude=0, tolerance=1.0)

# Select data at specific points
points = [(0, 0), (45, 45), (-45, -45)]
point_data = ds.hutil.select_at_points(points)

# Smart selection based on input types
equator_data = ds.hutil.select(lat=0)  # Select at latitude
meridian_data = ds.hutil.select(lon=0)  # Select at longitude
point_value = ds.hutil.select(lat=45, lon=45)  # Select at point
region_data = ds.hutil.select(lat=slice(0, 45), lon=slice(-45, 45))  # Select region
```

See the examples directory for more detailed examples.

## Documentation

For detailed documentation, see the [examples](./examples) directory and the docstrings in the code.

## Acknowledgements

This package was developed during [HK25](https://www.wcrp-esmo.org/activities/wcrp-global-km-scale-hackathon-2025) hackathon. It builds upon the excellent [easygems](https://easygems.org/) and [healpy](https://healpy.readthedocs.io/) packages for HEALPix functionality.

## License

MIT

## Author

Suvarchal K. Cheedela (suvarchal@duck.com)
