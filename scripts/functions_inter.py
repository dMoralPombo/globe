"""
This module contains functions for processing and analyzing Digital Elevation Models (DEMs) 
using various geospatial libraries. The main functionalities include clipping rasters, 
plotting rasters, applying bitmasks, warping and calculating statistics, and handling file
downloads and extractions. The module is designed to work with ArcticDEM data and includes
functions for processing tile intersections, calculating statistics, and stacking arrays.

To be called from calc_advanced.py.

Functions:
    process_tile_intersections(tile, strip_index_gdf, archdir):
        Processes tile intersections for ArcticDEM strips, downloading and unzipping files as needed.
    check_intersection(strip_geom, tile_coords, area_threshold=0.01):
        Checks if a strip geometry intersects with the tile coordinates and if the overlapping area exceeds a threshold.
    handle_strip_download(geocell, url, archdir):
        Downloads and unzips a StripDEM file if it does not already exist.
    handle_tile_download(tile):
        Downloads and unzips a tile shapefile if it does not already exist.
    download_strips(intersect_dems_df):
        Downloads all strips from a DataFrame of intersecting DEMs.
    download_file(geocell_folder, url):
        Downloads a file using wget.
    configuration():
        Reads configuration settings from a config file and returns relevant paths and parameters.
    intersection(supertile, subtile, strip_index_gdf, mosaic_index_gdf, archdir):
        Computes the intersection between the StripDEMs and the mosaic in a subtile.

@dmoralpombo (based in Jade Bowling's work)
"""

import os
import sys
import geopandas as gpd # type: ignore
from shapely.geometry import shape, mapping, Polygon, MultiPolygon, box # type: ignore
import numpy as np
from pygeotools.lib import timelib, iolib, malib, warplib, geolib # type: ignore
import seaborn as sns # type: ignore
from tqdm import tqdm # type: ignore
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
from osgeo import gdal # type: ignore
import rasterio as rio # type: ignore
from rasterio.mask import mask # type: ignore 
from rasterio.plot import plotting_extent, show # type: ignore
from rasterio.transform import from_origin # type: ignore
from rasterio.warp import calculate_default_transform, reproject, Resampling # type: ignore
from pyproj import Transformer, CRS # type: ignore
import fiona # type: ignore
from concurrent.futures import ThreadPoolExecutor 
import subprocess # type: ignore
import requests # type: ignore
import tarfile # type: ignore
import gzip # type: ignore
import shutil
import glob
import cartopy.crs as ccrs # type: ignore
import cartopy.feature as cfeature # type: ignore 
import pdemtools as pdem # type: ignore
import rioxarray as rxr # type: ignore
from affine import Affine # type: ignore
import gc
from mpl_toolkits.axes_grid1 import make_axes_locatable # type: ignore
from natsort import natsorted  # Natural sorting for  filenames # type: ignore
import contextily as cx # type: ignore
from geodatasets import get_path # type: ignore
from matplotlib.patches import Rectangle  # type: ignore
import configparser

# Define main directory
maindir = str("/media/luna/moralpom/globe/")
# and mosaic's directory
mosaic_dem = maindir + "data/ArcticDEM/mosaic/arcticdem_mosaic_100m_v4.1_dem.tif"
# Define spatial resolution of the strips
res = 2

def process_tile_intersections(
    tile,
    strip_index_gdf,
    archdir,
):
    """
    Processes tile intersections for ArcticDEM strips, downloading and unzipping files as needed.

    Parameters:
        tile (str): Identifier for the tile.
        strip_index_gdf (GeoDataFrame): Geopandas dataframe with ArcticDEM strip information.
        archdir (str): Archive directory for StripDEM files.

    Returns:
        tile_coords: Dataframe containing intersection data.
    """

    # Load the mosaic shapefile
    mosaic_index_path = "/media/luna/moralpom/globe/data/ArcticDEM/mosaic/ArcticDEM_Mosaic_Index_latest_shp/ArcticDEM_Mosaic_Index_v4_1_2m.shp"
    mosaic_index_gdf = gpd.read_file(mosaic_index_path)

    tile_row = mosaic_index_gdf[mosaic_index_gdf["tile"] == tile]
    tile_coords = tile_row.iloc[0]["geometry"]
    tile_bounds = tile_coords.bounds
    supertile = tile[:5]

    # Setup paths for the tile shapefile
    df_dir = os.path.join(
        maindir, f"data/ArcticDEM/ArcticDEM_stripfiles/{supertile}/df_csvs/"
    )
    os.makedirs(df_dir, exist_ok=True)
    intersection_df_path = os.path.join(df_dir, f"{tile}_df.csv")
    print(
        f"\n\n\n\nPART 1: FINDING INTERSECTING DEMS (for tile {tile})\ntile_bounds: {tile_bounds}.\n\n\n\n"
    )

    if os.path.exists(intersection_df_path):
        print(f"Loading existing intersection DataFrame from:\n {intersection_df_path}")
        return pd.read_csv(intersection_df_path)
    # else:
    print(f"Creating new intersection DataFrame:\n {intersection_df_path}")

    # Initialize variables
    lines = []
    n_intersections = 0

    # Process intersections
    for _, strip in tqdm(strip_index_gdf.iterrows(), total=strip_index_gdf.shape[0]):
        strip_geom = strip["geometry"]
        strip_name = strip["pairname"]
        geocell = strip["geocell"]
        url = strip["fileurl"]
        acqdate1 = strip["acqdate1"]

        if check_intersection(strip_geom, tile_coords):
            n_intersections += 1
            save_name = (
                "_".join([strip_name.split("_")[i] for i in [0, 2, 1, 3]]) + ".tif"
            )
            lines.append([tile, strip_name, acqdate1, save_name, geocell, url])
            handle_strip_download(geocell, url, archdir)

    # Create DataFrame and save results
    intersect_dems_df = pd.DataFrame(
        lines,
        columns=["Tile", "Strip_name", "Acq_date", "File_Name", "Geocell", "url"],
    ).drop_duplicates(subset=["File_Name"])

    print(f"\n\n\n\nPART 2: DOWNLOADING STRIPS... (for tile {tile}) \n\n\n\n")
    download_strips(intersect_dems_df)

    print(
        f"\n\n\n\nPART 3: ALL STRIPS DOWNLOADED for tile {tile}. CREATING intersect_dems_df.csv \n\n\n\n"
    )
    intersect_dems_df.to_csv(intersection_df_path, index=False)

    return intersect_dems_df


######################################################################


def check_intersection_og(strip_geom, tile_coords):
    """
    Checks if a strip geometry intersects with the tile coordinates.

    Parameters:
        strip_geom (Polygon or MultiPolygon): Geometry of the strip.
        tile_coords (Polygon): tile cell geometry.

    Returns:
        bool: True if intersects, False otherwise.
    """
    if isinstance(strip_geom, Polygon):
        return strip_geom.intersects(tile_coords)
    elif isinstance(strip_geom, MultiPolygon):
        return any(poly.intersects(tile_coords) for poly in strip_geom.geoms)
    return False

def check_intersection(strip_geom, tile_coords, area_threshold=0.01):
    """
    Checks if a strip geometry intersects with the tile coordinates and if
    the overlapping area exceeds a threshold.

    Parameters:
        strip_geom (Polygon or MultiPolygon): Geometry of the strip.
        tile_coords (Polygon): tile cell geometry.
        area_threshold (float): Minimum overlapping area (as a fraction of the tile area) 
                                required to include the strip.

    Returns:
        bool: True if intersects and the overlapping area meets the threshold, False otherwise.

    """
    tile_area = tile_coords.area  # Total area of the tile

    if isinstance(strip_geom, Polygon):
        overlap_geom = strip_geom.intersection(tile_coords)

        #return strip_geom.intersects(tile_coords)
    elif isinstance(strip_geom, MultiPolygon):
        overlap_geom = MultiPolygon(
            [poly.intersection(tile_coords) for poly in strip_geom.geoms]
        ).buffer(0)  # Buffer(0) fixes potential invalid geometries
        #return any(poly.intersects(tile_coords) for poly in strip_geom.geoms)
    else:
        return False
    
    # Calculate the overlapping area
    overlap_area = overlap_geom.area

    # Check if the overlapping area exceeds the threshold
    fraction = overlap_area / tile_area
    if fraction >= area_threshold:
        print(f"Overlap area: {overlap_area}, Fraction: {fraction}")
        return True
    else:
        return False

######################################################################


def handle_strip_download(geocell, url, archdir):
    """
    Downloads and unzips a StripDEM file if it does not already exist.

    Parameters:
        geocell (str): Geocell identifier.
        url (str): URL of the StripDEM file.
        archdir (str): Archive directory for StripDEM files.
    """
    fname = url.split("/")[-1]
    extracted_dir = os.path.join(archdir, geocell, fname)

    if os.path.exists(extracted_dir):
        # print(f"Strip already extracted: {extracted_dir}")
        return

    try:
        extracted_tar_path = extracted_dir[:-3]
        with gzip.open(extracted_dir, "rb") as f_gz_in:
            with open(extracted_tar_path, "wb") as f_tar_out:
                shutil.copyfileobj(f_gz_in, f_tar_out)

        with tarfile.open(extracted_tar_path, "r") as tar:
            tar.extractall(extracted_dir)
        os.remove(extracted_tar_path)
        print(f"Strip extracted: {fname}")

    except EOFError:
        print(f"Error: Corrupted or incomplete .gz file: {extracted_dir}")
        # Optionally, delete the corrupted file to allow re-download
        if os.path.exists(extracted_dir):
            os.remove(extracted_dir)
        print(f"Deleted corrupted file: {extracted_dir}")

    except FileNotFoundError:
        print(f"Strip not found: {extracted_dir}. Skipping.")


######################################################################


def handle_tile_download(tile):
    """
    Downloads and unzips a tile shapefile if it does not already exist.

    Parameters:
        tile (str): Tile identifier.
    """

    tile_id = tile + "_2m_v4.1"
    supertile = tile_id[:5]
    tile_fname_gz = f"{tile_id}.tar.gz"  # Modify based on naming convention
    tile_fname_dem = f"{tile_id}_dem.tif"  # Modify based on naming convention
    extracted_dir = f"/media/luna/archive/SATS/OPTICAL/ArcticDEM/ArcticDEM/tiles/{supertile}/{tile_id}"
    tilegz_path = os.path.join(
        extracted_dir,
        tile_fname_gz,
    )
    tiletar_path = tilegz_path[:-3]  # Removing '.gz'
    tiledem_path = os.path.join(
        extracted_dir,
        tile_fname_dem,
    )

    if os.path.exists(tiledem_path):
        print(f"Tile shapefile already extracted: {tiledem_path}")
        return
    if os.path.exists(tilegz_path):
        print(f"Found .gz file: {tilegz_path}. Extracting...")
        try:
            with gzip.open(tilegz_path, "rb") as f_gz_in:
                with open(tiletar_path, "wb") as f_tar_out:
                    shutil.copyfileobj(f_gz_in, f_tar_out)
            print(f"Extracted: {tiletar_path}")

            with tarfile.open(tiletar_path, "r") as tar:
                tar.extractall(extracted_dir)
            os.remove(tiletar_path)
            print(f"Extracted: {tile_fname_dem}")
            return
        except Exception as e:
            print(f"Error extracting .gz file: {e}")
            return

    # If .gz file is not found, download it
    url = f"https://data.pgc.umn.edu/elev/dem/setsm/ArcticDEM/mosaic/v4.1/2m/{supertile}/{tile_id}.tar.gz"
    print(f".gz file not found. Downloading from {url}...")
    try:
        subprocess.run(
            [
                "wget",
                "-r",
                "-N",
                "-nH",
                "-np",
                "-A",
                ".tar.gz",
                "--cut-dirs=3",
                "--no-host-directories",
                "-nd",
                "-P",
                extracted_dir,
                url,
            ],
            check=True,
        )
        print(f"Downloaded .gz file to {extracted_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading file: {e}")
        return

    # Extract the downloaded .gz file
    try:
        with gzip.open(tilegz_path, "rb") as f_gz_in:
            with open(tiletar_path, "wb") as f_tar_out:
                shutil.copyfileobj(f_gz_in, f_tar_out)
        print(f"Extracted tar file: {tiletar_path}")

        with tarfile.open(tiletar_path, "r") as tar:
            tar.extractall(extracted_dir)
        os.remove(tiletar_path)
        print(f"Extracted DEM file: {tiledem_path}")
    except Exception as e:
        print(f"Error extracting downloaded .gz file: {e}")


######################################################################


def download_strips(intersect_dems_df):

    ########################################################################
    # PART 3: download all strips from CSV
    ########################################################################
    download_folder = (
        "/media/luna/archive/SATS/OPTICAL/ArcticDEM/ArcticDEM/strips/s2s041/2m"
    )

    # Load the list of URLs from the CSV file
    # Column name in the CSV containing the URLs, file names and geocell
    url_column = "url"
    # filename_column = "File_Name"
    geocell_column = "Geocell"

    # Create a list of URLs for files that are missing, with subfolders by geocell
    missing_files = []
    for _, row in tqdm(intersect_dems_df.iterrows()):
        geocell_folder = os.path.join(download_folder, row[geocell_column])
        os.makedirs(
            geocell_folder, exist_ok=True
        )  # Create geocell subfolder if it doesn't exist
        # filepath = os.path.join(geocell_folder, row[filename_column])
        unzipped_name = row[url_column].split("/")[-1]
        unzfilepath = os.path.join(geocell_folder, unzipped_name)

        # Add to the missing files list if the file doesn't exist
        # if not os.path.exists(filepath) and not os.path.exists(unzfilepath):
        if not os.path.exists(unzfilepath):
            missing_files.append((geocell_folder, row[url_column]))

    print(f"{len(missing_files)} files missing.")
    ################################################################################
    # Use ThreadPoolExecutor to download files concurrently
    max_workers = 24  # Number of concurrent downloads

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(
            total=len(missing_files), desc="Downloading files", unit="file"
        ) as pbar:

            # Submit a download task for each missing file
            futures = [
                executor.submit(download_file, geocell_folder, url)
                for geocell_folder, url in missing_files
            ]

            # Wait for all tasks to complete
            for future in futures:
                try:
                    future.result()  # This will raise any exception if occurred during download
                except Exception as e:
                    print(f"Error occurred during download: {e}")
                finally:
                    pbar.update(1)  # Move progress bar forward

    missing_files = []
    del missing_files
    gc.collect()

######################################################################

# Function to download a file using wget
def download_file(geocell_folder, url):
    # Ensure the geocell folder exists before attempting download
    os.makedirs(geocell_folder, exist_ok=True)
    try:
        # Run wget in quiet mode (-q) to reduce verbosity
        subprocess.run(
            [
                "wget",
                "-r",
                "-N",
                "-nH",
                "-np",
                "-q",
                "-A",
                ".tar.gz",
                "--cut-dirs=3",
                "--no-host-directories",
                "-nd",
                "-P",
                geocell_folder,
                url,
            ],
            check=True,
        )
        print(f"Downloaded to {geocell_folder}\n from URL {url}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to download: {url}, error: {e}")

######################################################################
def configuration():
    
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
    mosaic_index_path = "/media/luna/moralpom/globe/data/ArcticDEM/mosaic/ArcticDEM_Mosaic_Index_latest_shp/ArcticDEM_Mosaic_Index_v4_1_2m.shp"
    # mosaic_index_path = f"/media/luna/archive/SATS/OPTICAL/ArcticDEM/ArcticDEM/mosaic/v4.1/2m/{supertile}/index/{tile}_2m_v4.1_index.shp"
    mosaic_index_gdf = gpd.read_file(mosaic_index_path)
    
    return maindir, archdir, res, diffndv, strip_index_gdf, mosaic_index_gdf, stats_columns

######################################################################
def intersection(supertile, subtile, strip_index_gdf, mosaic_index_gdf, archdir):
    """
    Computes the intersection between a the StripDEMs and the mosaic in a subtile.

    Args:
        supertile (str): Identifier for the supertile.
        subtile (str): Identifier for the subtile.
        strip_index_gdf (GeoDataFrame): GeoDataFrame containing strip index information.
        mosaic_index_gdf (GeoDataFrame): GeoDataFrame containing mosaic index information.

    Returns:
        tuple: Contains the tile identifier, tile coordinates, tile bounds, and DataFrame of intersecting DEMs.
    """

    tile = f"{supertile}_{subtile}"
    tile_id = tile + "_2m_v4.1"

    # Temporary files for the clipped mosaic/stripDEMs
    mosaic_dir = maindir + f"data/ArcticDEM/mosaic/temp/{tile}/"
    strips_dir = maindir + f"data/ArcticDEM/temp2/{tile}/"

    try:
        tile_row = mosaic_index_gdf[mosaic_index_gdf["dem_id"] == tile_id]
    except KeyError:
        tile_row = mosaic_index_gdf[mosaic_index_gdf["DEM_ID"] == tile_id]

    if tile_row.shape[0] == 1:
        print(f"Processing tile: {tile}\n\n\n\n\n\n")

        # Get the bounds of the tile cell
        tile_coords = tile_row.iloc[0]["geometry"]
        tile_bounds = tile_coords.bounds

        # Crop the tile bounds to avoid overlapping between tiles
        tile_bounds_mod = (
            tile_bounds[0] + 100,
            tile_bounds[1] + 100,
            tile_bounds[2] - 100,
            tile_bounds[3] - 100,
        )
        tile_bounds = tile_bounds_mod
        #################################################################
        # This section imports the list of "intersecting" StripDEMs
        # (or creates it, if it had not been already done)

        intersect_dems_df = process_tile_intersections(
            tile=tile,
            strip_index_gdf=strip_index_gdf,
            archdir=archdir,
        )

        if intersect_dems_df.empty:
            print("intersect_dems_df is empty (No StripDEMs in this tile).\
                    \n Skip")
            return None
        else:
            print("intersect_dems_df loaded - NOT empty")

    else:
        print(f"Tile {tile} not found in the mosaic index. Skipping...")
    
    return tile, tile_coords, tile_bounds, intersect_dems_df