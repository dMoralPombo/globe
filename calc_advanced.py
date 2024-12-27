"""
This is a script to calculate STD maps of a tile including all
intersecting DEMs whose STD is less than a threshold value.
The script will download the DEMs, clip them to the tile bounds,
and calculate the difference between the DEM and the reference DEM (mosaic), 
and then compute statistics on the differences.
"""

import os
import configparser
import glob
import gc
import geopandas as gpd # type: ignore
from tqdm import tqdm # type: ignore
import pandas as pd # type: ignore
from affine import Affine # type: ignore
 
# Define main directory
maindir = str("/media/luna/moralpom/globe/")
archdir = str("/media/luna/archive/SATS/OPTICAL/ArcticDEM/ArcticDEM/strips/s2s041/2m/")
os.chdir(maindir + "github_ready/globe/")

from functions_std import *
from functions_std import (
    process_tile_intersections,
    clip_raster_to_cell,
    reduce_strip,
    stack,
)

# Download only or full processing pipeline?
download_only = not bool(input("Download only? (Enter for YES)   "))

# Read config
config = configparser.ConfigParser()
config.read("config.cfg")

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
# Convert stats_columns to list of strings without whitespace or "\n" or quotes:
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
os.chdir(maindir + "code/jade/")

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
supertiles = input("Enter the supertile(s) to process (e.g. '22_31, 22_32'): ").split(",")
supertiles = [supertile.strip() for supertile in supertiles]

subtiles = ["1_1", "1_2", "2_1", "2_2"]

# Define the threshold for filtering out StripDEMs by std
threshold = float(input('Enter the threshold for filtering out StripDEMs by std: '))
if threshold = None:
    threshold = 20 # Default value

for supertile in supertiles:
    for subtile in subtiles:
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
                break
            else:
                print("intersect_dems_df loaded - NOT empty")

            if download_only is True:
                continue
            else:
                # Set file path for tile shapefile and dataframe storage
                df_dir = (
                    maindir
                    + f"data/ArcticDEM/ArcticDEM_stripfiles/{supertile}/df_csvs/"
                )

                # Ensure the dataframe directory exists
                if not os.path.exists(df_dir):
                    os.makedirs(df_dir)

                # Set file path for mosaic DEM
                mosaic_dem = (
                    maindir + "data/ArcticDEM/mosaic/arcticdem_mosaic_100m_v4.1_dem.tif"
                )
                mosaic_dem = f"/media/luna/archive/SATS/OPTICAL/ArcticDEM/ArcticDEM/mosaic/v4.1/2m/{supertile}/{tile_id}_dem.tif"
                diffndv = -9999

                # Clip the mosaic DEM to the tile bounds
                mosaic_clipped_fn = clip_raster_to_cell(
                    mosaic_dem, tile_bounds, mosaic_dir, diffndv
                )
                print("mosaic clipped to tile_bounds")

                intersect_dems_df = intersect_dems_df.sort_values(
                    "Acq_date"
                )  # Sort dataframe in ascending date order
                intersect_dems_df.reset_index(inplace=True)  # Reset the index
                sorted_savename_nofilt = intersect_dems_df[
                    "Strip_name"
                ].tolist()  # Make list of sorted filenames
                sorted_geocell_nofilt = intersect_dems_df[
                    "Geocell"
                ].tolist()  # Make list of sorted geocells
                date_nofilt = intersect_dems_df[
                    "Acq_date"
                ].tolist()  # Make list of sorted dates

                ##############################################################################
                # This section imports georeferenced DSM, resamples the georeferenced DSM
                # (to match reference DSM - ArcticDEM 100m mosaic),
                # calculates difference between reference DSM, and exports important statistics

                # Loading stats_file
                output_stats_file = df_dir + str(tile) + "_stats_df_2m_gbmod.csv"

                # Try to access diff_stats CSV if it was created already:
                if os.path.exists(output_stats_file):
                    print(
                        f"Loading existing Statistics DataFrame from \n{output_stats_file}"
                    )
                    df_stats = pd.read_csv(output_stats_file)
                    df_stats = df_stats.drop_duplicates(subset=["filename"])
                    df_stats = df_stats.sort_values(
                        "acqdate"
                    )  # Sort dataframe in ascending date order

                    # Find where to restart in sorted_savename_nofilt
                    last_stored = df_stats["filename"].iloc[-1]
                    try:
                        start_index = (
                            sorted_savename_nofilt.index(last_stored) + 1
                        )  # Continue from next file
                        print(f"Resuming from index {start_index}...")
                    except ValueError:
                        print(
                            "Last processed strip not found in the list. Starting from scratch."
                        )
                        start_index = 0
                else:
                    print("No existing DataFrame found, processing from scratch...")
                    df_stats = pd.DataFrame(
                        columns=stats_columns
                    )  # Create a new DataFrame with the defined columns
                    # Create empty lists for the stack and statistics
                    start_index = 0

                print(
                    "---- " + str(len(sorted_savename_nofilt)) + " files"
                )  # len = length
                temp_dir_filledarrays = f"/media/luna/moralpom/globe/data/ArcticDEM/temp/filled_arrays/{tile}/masked/"
                os.makedirs(temp_dir_filledarrays, exist_ok=True)

                for item, strip_name in tqdm(
                    enumerate(sorted_savename_nofilt[start_index:], start=start_index)
                ):
                    # Check if strip_name already exists in df_stats to skip reprocessing
                    if strip_name in df_stats["filename"].values:
                        print(f"Skipping {strip_name}, already processed.")
                        continue  # Skip to the next strip if already processed
                    else:
                        try:
                            geocell_i = sorted_geocell_nofilt[item]
                            date_nofilt_i = date_nofilt[item]

                            print(
                                str(item)
                                + "/"
                                + str(len(sorted_savename_nofilt))
                                + " files"
                            )
                            print(strip_name)

                            diff_stats, mean_r2, rmse, proc_file = reduce_strip(
                                strip_name,
                                tile_bounds,
                                strips_dir,
                                mosaic_clipped_fn,
                                geocell_i,
                                diffndv,
                                plotting=False,
                                temp_dir=temp_dir_filledarrays,
                            )

                            if (
                                type(diff_stats) is tuple
                            ):  # In case there is overlap (most of the times)
                                new_row_stats = pd.DataFrame(
                                    [
                                        [
                                            strip_name,
                                            date_nofilt_i,
                                            geocell_i,
                                            diff_stats[0],
                                            mean_r2,
                                            diff_stats[1],
                                            diff_stats[2],
                                            diff_stats[3],
                                            diff_stats[4],
                                            diff_stats[5],
                                            diff_stats[6],
                                            diff_stats[7],
                                            diff_stats[8],
                                            diff_stats[9],
                                            diff_stats[10],
                                            diff_stats[11],
                                            diff_stats[12],
                                            diff_stats[13],
                                            rmse,
                                        ]
                                    ],
                                    columns=df_stats.columns,
                                )
                                df_stats = pd.concat(
                                    [df_stats, new_row_stats], ignore_index=True
                                )

                                # Save to DataFrame periodically (every 10 strips, for example)
                                if item % 10 == 0:
                                    df_stats.to_csv(
                                        output_stats_file, mode="w", index=False
                                    )  # Save the updated DataFrame to the CSV file
                                    print(
                                        f"Checkpoint: Saved progress at {strip_name}."
                                    )

                        except IndexError:
                            print("\nFile not downloaded yet... Continue (/) \n")
                            break

                        # Cleaning the variables for memory:
                        diff_stats = []
                        rmse = []
                        mean_r2 = []
                        new_row_stats = []

                        del diff_stats, rmse, mean_r2, new_row_stats

                # Save remaining data to CSV file after loop ends
                df_stats.to_csv(output_stats_file, mode="w", index=False)
                print("df_stats saved definitely. \n\n ",
                      df_stats)
                
                # Filter DSMs that are less than threshold = 50 sigma dh (originally it was 20 sigma)
                spread_filtered_df = df_stats[df_stats["std_dh"] < threshold]
                spread_filtered_df = spread_filtered_df.drop_duplicates()
                print(f"Length of spread_filtered_df_20: {len(spread_filtered_df)}")

                all_arrays = glob.glob(
                    os.path.join(temp_dir_filledarrays, "masked_arr_W*.npy")
                )

                # Get the core filenames from the dataframe
                valid_cores = spread_filtered_df["filename"].tolist()

                # Filter files in the directory
                stackarrays = [
                    file
                    for file in all_arrays
                    if os.path.basename(file)
                    .removeprefix("masked_arr_")
                    .removesuffix(".npy")
                    in valid_cores
                ]

                if stackarrays:
                    stack(
                        stackarrays,
                        tile,
                        tile_bounds,
                        "no",
                    )
                else:
                    print("stackarrays is empty")

                print("\n\n\nStack run. Proceeding to next tile...\n\n\n")
                gc.collect()

# Make sure to delete the temporary files after processing
if os.path.exists(mosaic_clipped_fn):
    os.remove(mosaic_clipped_fn)

# Explicitly call garbage collector to free up memory
gc.collect()
