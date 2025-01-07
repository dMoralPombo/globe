# tests/test_functions_std.py
import sys
import os
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import pytest
from unittest.mock import patch, MagicMock
from scripts.functions_std import *

@patch("scripts.functions_std.configparser.ConfigParser")
@patch("scripts.functions_std.gpd.read_file")
def test_configuration(mock_read_file, mock_config_parser):
    # Mock configparser behavior
    mock_config = MagicMock()
    mock_config.get.side_effect = lambda section, key: {
        ("paths", "maindir"): "/media/luna/moralpom/globe/",
        ("paths", "archdir"): "/media/luna/archive/SATS/OPTICAL/ArcticDEM/ArcticDEM/strips/s2s041/2m/",
        ("paths", "supertile_dir"): "some_supertile_dir",
        ("paths", "stripfiles_dir"): "some_stripfiles_dir",
        ("region", "region_name"): "some_region",
        ("region", "supertile_id"): "supertile_001",
        ("region", "tile_id"): "tile_001",
        ("tile", "grid_shapefile"): "some_grid_shapefile",
        ("tile", "df_dir"): "some_df_dir",
        ("strip", "url_template"): "http://example.com/{strip}",
        ("stats", "stats_columns"): "filename,acqdate,geocell,count,mean,min_dh,max_dh,mean_dh,std_dh,med_dh,mad_dh,q1_dh,q2_dh,iqr_dh,mode_dh,p16_dh,p84_dh,spread_dh,rmse_dh",
    }.get((section, key), "")
    mock_config_parser.return_value = mock_config

    # Mock geopandas.read_file and .to_crs()
    mock_gdf = MagicMock()
    mock_read_file.return_value = mock_gdf
    mock_gdf.to_crs.return_value = mock_gdf  # Ensure .to_crs() returns the mock_gdf itself

    # Call the function
    maindir, archdir, res, diffndv, strip_index_gdf, mosaic_index_gdf, stats_columns = configuration()

    # Assertions
    assert maindir == "/media/luna/moralpom/globe/"
    assert archdir == "/media/luna/archive/SATS/OPTICAL/ArcticDEM/ArcticDEM/strips/s2s041/2m/"
    assert res == 2
    assert diffndv == -9999
    assert stats_columns == [
        "filename",
        "acqdate",
        "geocell",
        "count",
        "mean",
        "min_dh",
        "max_dh",
        "mean_dh",
        "std_dh",
        "med_dh",
        "mad_dh",
        "q1_dh",
        "q2_dh",
        "iqr_dh",
        "mode_dh",
        "p16_dh",
        "p84_dh",
        "spread_dh",
        "rmse_dh",
    ]
    assert strip_index_gdf is mock_gdf  # This should now work

@patch("scripts.functions_std.process_tile_intersections")
def test_intersection(mock_process_tile_intersections):
    # Mock inputs
    supertile = "N72E18"
    subtile = "01"
    tile = f"{supertile}_{subtile}"
    tile_id = f"{tile}_2m_v4.1"

    mock_strip_index_gdf = MagicMock()
    mock_mosaic_index_gdf = gpd.GeoDataFrame(
        {
            "dem_id": [tile_id],
            "geometry": [Polygon([(0, 0), (0, 10), (10, 10), (10, 0)])]
        }
    )
    
    # Mock process_tile_intersections return value
    mock_process_tile_intersections.return_value = pd.DataFrame({
        "Tile": [tile],
        "Strip_name": ["strip1"],
        "Acq_date": ["2024-01-01"],
        "File_Name": ["file1.tif"],
        "Geocell": ["cell1"],
        "url": ["http://example.com/file1"]
    })

    # Call the function
    result = intersection(supertile, subtile, mock_strip_index_gdf, mock_mosaic_index_gdf, "/path/to/archdir")

    # Assertions
    assert result is not None
    assert len(result) == 4  # tile, tile_coords, tile_bounds, intersect_dems_df
    assert result[0] == tile
    assert isinstance(result[1], Polygon)
    assert isinstance(result[2], tuple)
    assert not result[3].empty

@patch("scripts.functions_std.check_intersection")
@patch("scripts.functions_std.handle_strip_download")
@patch("scripts.functions_std.gpd.read_file")
@patch("os.path.exists")
@patch("pandas.DataFrame.to_csv")
@patch("pandas.read_csv")
@patch("scripts.functions_std.check_intersection")
@patch("scripts.functions_std.handle_strip_download")
@patch("scripts.functions_std.gpd.read_file")
@patch("os.path.exists")
@patch("pandas.DataFrame.to_csv")
def test_process_tile_intersections(
    mock_read_csv, mock_to_csv, mock_exists, mock_read_file, mock_handle_strip_download, mock_check_intersection
):
    # Mock inputs
    tile = "22_38_1_1"
    mock_strip_index_gdf = gpd.GeoDataFrame(
        {
            "geometry": [Polygon([(0, 0), (0, 10), (10, 10), (10, 0)])],
            "pairname": ["strip1"],
            "geocell": ["cell1"],
            "fileurl": ["http://example.com/file1"],
            "acqdate1": ["2024-01-01"]
        }
    )
    mock_read_file.return_value = gpd.GeoDataFrame({
        "tile": [tile],
        "geometry": [Polygon([(0, 0), (0, 10), (10, 10), (10, 0)])]
    })
    mock_exists.side_effect = lambda path: path.endswith(".csv")  # Assume CSV exists
    mock_check_intersection.return_value = True
    mock_read_csv.return_value = pd.DataFrame({"dummy_column": []})  # Mock CSV read

    # Call the function
    result = process_tile_intersections(tile, mock_strip_index_gdf, "/path/to/archdir")

    # Assertions
    mock_read_csv.assert_called_once()  # Ensure read_csv is called
    assert result is not None  # Validate function output

@patch("os.path.exists")
@patch("gzip.open")
@patch("tarfile.open")
@patch("gzip.open")
def test_handle_strip_download(mock_gzip_open, mock_tarfile_open, mock_exists):
    # Ensure mock setup is correct
    mock_exists.return_value = True
    mock_tarfile_open.return_value.__enter__.return_value.extractall.return_value = None
    mock_gzip_open.return_value = mock.Mock()  # If needed, mock the return value here

    # Call the function
    handle_strip_download("cell1", "http://example.com/file1.tar.gz", "/path/to/archdir")

    # Assert gzip.open was called
    mock_gzip_open.assert_called_once()

def test_check_intersection():
    # Test case: Polygon intersects
    strip_geom = Polygon([(0, 0), (0, 10), (10, 10), (10, 0)])
    tile_coords = Polygon([(5, 5), (5, 15), (15, 15), (15, 5)])
    assert check_intersection(strip_geom, tile_coords)

    # Test case: Polygon does not intersect
    strip_geom = Polygon([(20, 20), (20, 30), (30, 30), (30, 20)])
    assert not check_intersection(strip_geom, tile_coords)

    # Test case: MultiPolygon intersects
    strip_geom = MultiPolygon([
        Polygon([(20, 20), (20, 30), (30, 30), (30, 20)]),
        Polygon([(5, 5), (5, 15), (15, 15), (15, 5)])
    ])
    assert check_intersection(strip_geom, tile_coords)

    # Test case: MultiPolygon does not intersect
    strip_geom = MultiPolygon([
        Polygon([(20, 20), (20, 30), (30, 30), (30, 20)])
    ])
    assert not check_intersection(strip_geom, tile_coords)
