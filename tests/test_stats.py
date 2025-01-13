import sys
import os
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon
import shutil
import glob
import numpy as np
import rasterio as rio
from rasterio.coords import BoundingBox
from rasterio.transform import Affine

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import pytest
from unittest.mock import patch, MagicMock
from scripts.functions_all import *


# Test for clip_raster_to_cell
@patch("rasterio.open")
@patch("os.path.exists")
@patch("os.makedirs")
def test_clip_raster_to_cell(mock_makedirs, mock_exists, mock_rio_open):
    # Mock inputs
    raster = "/path/to/raster.tif"
    bounds = (10, 10, 20, 20)
    output_dir = "/path/to/output"

    # Mock behaviors
    mock_exists.side_effect = lambda path: path == raster
    mock_rio_open.return_value.__enter__.return_value.read.return_value = np.array([[1, 2], [3, 4]])
    mock_rio_open.return_value.__enter__.return_value.bounds = BoundingBox(5, 5, 25, 25)
    # mock_rio_open.return_value.__enter__.return_value.transform = "transform"
    mock_rio_open.return_value.__enter__.return_value.transform = Affine(2.0, 0.0, 0.0, 0.0, -2.0, 0.0)

    # Call the function
    clipped_fn = clip_raster_to_cell(raster, bounds, output_dir)

    # Assertions
    mock_exists.assert_called_with(raster)
    mock_rio_open.assert_called_once_with(raster)
    mock_makedirs.assert_any_call(output_dir, exist_ok=True)
    assert clipped_fn is not None
    assert clipped_fn.startswith(output_dir)


@patch("glob.glob")
@patch("gzip.open")
@patch("shutil.copyfileobj")
@patch("tarfile.open")
def test_find_and_unzip(mock_tarfile_open, mock_copyfileobj, mock_gzip_open, mock_glob):
    # Mock inputs
    pathfile = "/path/to/file"

    # Mock behaviors
    mock_glob.side_effect = lambda pattern: ["/path/to/file_dem.tif"] if pattern.endswith("*.tif") else []
    mock_gzip_open.return_value.__enter__.return_value = MagicMock()
    mock_tarfile_open.return_value.__enter__.return_value.extractall.return_value = None

    # Call the function
    result = find_and_unzip(pathfile)

    # Assertions
    mock_glob.assert_called_with('/path/to/file*.gz')
    mock_gzip_open.assert_not_called()  # No .gz files should be processed
    assert result == "/path/to/file_dem.tif"
