"""
Calculate STD maps of a tile including all intersecting DEMs whose STD 
is less than a threshold value, using the functions from functions_std.py.

The script will download the DEMs, clip them to the tile bounds, calculate
the difference between the DEM and the reference DEM (mosaic), and then 
compute statistics on the differences.

Parameters:
    download_only (bool): Flag to determine if only downloading is required.
    supertiles (list): List of supertiles to process.
    subtiles (list): List of subtiles to process.
    threshold (float): Threshold for filtering out StripDEMs by std.
    

Returns:
    rasters (GeoTIFF): Rasters of the STD maps for each tile, number of DEMs and
    average elevation, and its corresponding JPEG images.   
"""

import os
import glob
import gc
import geopandas as gpd # type: ignore
from tqdm import tqdm # type: ignore
import pandas as pd # type: ignore
from affine import Affine # type: ignore
from functions_std import *
from functions_std import (
    process_tile_intersections,
    clip_raster_to_cell,
    reduce_strip,
    stack,
)

# Download only or full processing pipeline?
download_only = not bool(input("Download only? (Enter for YES)   "))

# Load initial variables (directories, resolution, indexes, stats columns_names)
maindir, archdir, res, diffndv, strip_index_gdf, mosaic_index_gdf, stats_columns = configuration()

# Define the tiles to process
supertiles = input("Enter the supertile(s) to process (e.g. '22_31, 22_32'): ").split(",")
supertiles = [supertile.strip() for supertile in supertiles]
subtiles = ["1_1", "1_2", "2_1", "2_2"]

# Define the threshold for filtering out StripDEMs by std
try:
    threshold = float(input('Enter the threshold for filtering out StripDEMs by std: '))
    if not threshold:
        raise ValueError
except ValueError:
    print("Invalid input. Using default threshold value of 20.")
    threshold = 20  # Default value

for supertile in supertiles:
    for subtile in subtiles:
        # intersection(supertile, subtile, strip_index_gdf, mosaic_index_gdf, archdir)
        tile = f"{supertile}_{subtile}"
        tile_id = tile + "_2m_v4.1"

        # Temporary files for the clipped mosaic/stripDEMs
        mosaic_dir = maindir + f"data/ArcticDEM/mosaic/temp/{tile}/"
        strips_dir = maindir + f"data/ArcticDEM/temp2/{tile}/"

        # Load the intersecting DEMs dataframe
        tile, tile_coords, tile_bounds, intersect_dems_df = intersection(supertile, subtile, strip_index_gdf, mosaic_index_gdf, archdir)
        if download_only is False:
            df_stats = stats_calculator(supertile, subtile, tile_bounds, intersect_dems_df, strip_index_gdf, mosaic_index_gdf, mosaic_dir, strips_dir, stats_columns)
            
            # Stacking
            stackador(df_stats, threshold, tile, tile_bounds)
            print("\n\n\nStack run. Proceeding to next tile...\n\n\n")
            gc.collect()


# Explicitly call garbage collector to free up memory
gc.collect()
