#!/usr/bin/env python
# coding: utf-8

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="hutil",
    version="0.1.0",
    author="Suvarchal K. Cheedela",
    author_email="suvarchal@duck.com",
    description="HEALPix utilities for xarray datasets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/suvarchal/hutil",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Topic :: Scientific/Engineering :: GIS",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "xarray",
        "healpy",
        "matplotlib",
        "cartopy",
        "shapely",
    ],
    extras_require={
        "shapefile": ["geopandas"],
    },
)
