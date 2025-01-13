"""
This module contains functions for processing and analyzing Digital Elevation Models (DEMs)
using various geospatial libraries. The main functionalities include clipping rasters,
plotting rasters, applying bitmasks, warping and calculating statistics, and handling file
downloads and extractions. The module is designed to work with ArcticDEM data and includes
functions for processing tile intersections, calculating statistics, and stacking arrays.

To be called from calc_advanced.py.

Functions:
    configuration():
        Reads configuration settings from a config file and returns relevant paths and parameters.

@dmoralpombo (based in Jade Bowling's work)
"""

import geopandas as gpd  # type: ignore
import configparser

# Define main directory
maindir = str("/media/luna/moralpom/globe/")
# and mosaic's directory
mosaic_dem = maindir + "data/ArcticDEM/mosaic/arcticdem_mosaic_100m_v4.1_dem.tif"
# Define spatial resolution of the strips
res = 2

######################################################################


def configuration():
    """
    This function reads a configuration file to retrieve various paths and parameters required for processing.
    It also loads and reprojects shapefiles for further use.

        maindir (str): Main directory path.
        archdir (str): Archive directory path.
        res (int): Spatial resolution of the strips.
        diffndv (int): No data value for difference calculations.
        strip_index_gdf (GeoDataFrame): GeoDataFrame containing the strip index shapefile data.
        mosaic_index_gdf (GeoDataFrame): GeoDataFrame containing the mosaic index shapefile data.
        stats_columns (list): List of column names for statistics.
    """
    # Read config
    config = configparser.ConfigParser()
    config.read("/media/luna/moralpom/globe/github_ready/globe/config.cfg")

    # Retrieve paths
    maindir = config.get("paths", "maindir")
    archdir = config.get("paths", "archdir")
    supertile_dir = config.get("paths", "supertile_dir")
    stripfiles_dir = config.get("paths", "stripfiles_dir")

    # Retrieve region-specific values
    region_name = config.get("region", "region_name")
    supertile = config.get("region", "supertile_id")
    tile_id = config.get("region", "tile_id")

    # Retrieve tile and strip-related values
    grid_shapefile = config.get("tile", "grid_shapefile")
    df_dir = config.get("tile", "df_dir")
    url_template = config.get("strip", "url_template")

    # Retrieve stats column names
    stats_columns = config.get("stats", "stats_columns").split(",")
    stats_columns = [column.strip() for column in stats_columns]

    # Print to verify
    print(f"Main Directory: {maindir}")
    print(f"Region: {region_name}")
    # print(f"SuperTile ID: {supertile}")
    print(f"Grid Shapefile: {grid_shapefile}")
    print(f"Output Directory for CSVs: {df_dir}")
    print(f"URL Template: {url_template}")

    # Define main directory
    maindir = str("/media/luna/moralpom/globe/")
    archdir = str("/media/luna/archive/SATS/OPTICAL/ArcticDEM/ArcticDEM/strips/s2s041/2m/")

    # Define spatial resolution of the strips
    res = 2

    # No data value, 0 will give odd outputs
    diffndv = -9999

    # Load the stripfile shapefile
    strip_index_path = (
        maindir
        + "data/ArcticDEM/ArcticDEM_Strip_Index_latest_shp/ArcticDEM_Strip_Index_s2s041.shp"
    )
    strip_index_gdf = gpd.read_file(strip_index_path)
    # Reproject the strip GeoDataFrame to the desired CRS (EPSG:3413)
    strip_index_gdf = strip_index_gdf.to_crs("EPSG:3413")

    # Load the mosaic shapefile
    mosaic_index_path = maindir + "/data/ArcticDEM/mosaic/ArcticDEM_Mosaic_Index_latest_shp/ArcticDEM_Mosaic_Index_v4_1_2m.shp"
    # mosaic_index_path = f"/media/luna/archive/SATS/OPTICAL/ArcticDEM/ArcticDEM/mosaic/v4.1/2m/{supertile}/index/{tile}_2m_v4.1_index.shp"
    mosaic_index_gdf = gpd.read_file(mosaic_index_path)

    return maindir, archdir, res, diffndv, strip_index_gdf, mosaic_index_gdf, stats_columns
