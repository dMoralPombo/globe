"""
This module contains functions for processing and analyzing Digital Elevation Models (DEMs) 
using various geospatial libraries. The main functionalities include clipping rasters, 
plotting rasters, applying bitmasks, warping and calculating statistics, and handling file
downloads and extractions. The module is designed to work with ArcticDEM data and includes
functions for processing tile intersections, calculating statistics, and stacking arrays.

To be called from calc_advanced.py.

Functions:
    clip_raster_to_cell(raster, bounds, output_dir, nodata_value):
        Clips a raster to a specified tile cell defined by its bounds.
    plot_clipped_rasters(raster1, raster2, bounds, title):
        Plots two clipped rasters side by side with tile bounds overlaid.
    find_and_unzip(pathfile):
        Finds a .tif file and/or unzips the .gz file if necessary.
    warp_and_calculate_stats(mosaic_clipped_fn, stripdem_clipped_fn):
        Warps rasters to the same resolution, extent, and projection, and calculates statistics.
    reduce_strip(strip_name, tile_bounds, strips_dir, mosaic_clipped_fn, geocell_i, diffndv, plotting, temp_dir):
        Processes a strip DEM to match the spatial extent of a fixed array, applies a bitmask, saves the result, and returns statistics.
    calculate_statistics(running_sum, running_squared_sum, valid_count):
        Calculates mean and standard deviation from running totals.
    stats_calculator(supertile, subtile, tile_bounds, intersect_dems_df, strip_index_gdf, mosaic_index_gdf, mosaic_dir, strips_dir, stats_columns):
        Computes statistics for a tile and saves the results.

@dmoralpombo (based in Jade Bowling's work)
"""
import os
import sys
import geopandas as gpd # type: ignore
from shapely.geometry import box # type: ignore
import numpy as np
from pygeotools.lib import iolib, malib, warplib # type: ignore
from tqdm import tqdm # type: ignore
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import rasterio as rio # type: ignore
from rasterio.windows import from_bounds # type: ignore
import gzip # type: ignore
import shutil
import glob
import tarfile # type: ignore
import gc

# Define main directory
maindir = str("/media/luna/moralpom/globe/")
# and mosaic's directory
mosaic_dem = maindir + "data/ArcticDEM/mosaic/arcticdem_mosaic_100m_v4.1_dem.tif"
# Define spatial resolution of the strips
res = 2


##########################################################################################


def clip_raster_to_cell(raster, bounds, output_dir, nodata_value=-9999):
    """
    This is a function to clip (crop) a raster to a tile cell (defined by its bounds)

    Args:
        raster (str): Path to the input raster file.
        bounds (tuple): Bounds of the tile cell to clip the raster to (xmin, ymin, xmax, ymax).
        output_dir (str): Directory to store the output clipped raster.
        nodata_value (int, optional): Value to be ignored as "no data". Defaults to -9999.
    Returns:
        clipped_fn (str): Path to the clipped raster file, or None if no overlap.
    """
    temp_dir = "/media/luna/moralpom/globe/data/ArcticDEM/temp/"
    outputdir2 = "/media/luna/moralpom/globe/data/ArcticDEM/temp2/"

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(outputdir2, exist_ok=True)
    # Ensure output directories exist
    for directory in [output_dir, temp_dir]:
        try:
            os.makedirs(directory, exist_ok=True)
        except Exception as e:
            print(f"Error creating directory {directory}: {e}")
            return None

    if not os.path.exists(raster):
        print(f"Unzipping raster:\n {raster}")
        raster_n = raster[:-8]
        raster = find_and_unzip(raster_n)

    try:
        # Open the raster file
        with rio.open(raster) as src:
            # Check if bounds overlap with raster's extent
            src_bounds = src.bounds
            print(f"Raster bounds: {src_bounds}")
            # print(f"Clipping bounds: {bounds}")

            # Check for overlap
            if (
                bounds[2] <= src_bounds.left
                or bounds[0] >= src_bounds.right
                or bounds[3] <= src_bounds.bottom
                or bounds[1] >= src_bounds.top
            ):
                print("No overlap between raster and bounds.")
                return None  # No overlap, so exit the function

            # Create a window using the tile bounds
            window = rio.windows.from_bounds(*bounds, transform=src.transform)

            # Adjust window if it exceeds raster dimensions
            window = window.intersection(
                rio.windows.Window(
                    col_off=0, row_off=0, width=src.width, height=src.height
                )
            )

            print(f"Window to be used for clipping raster: {window}")

            # Read the clipped data within the window
            clipped_data = src.read(1, window=window)

            # Mask the NoData value (-9999) in the clipped data
            masked_data = np.ma.masked_equal(clipped_data, nodata_value)

            # Update metadata for the clipped raster
            clipped_meta = src.meta.copy()
            clipped_meta.update(
                {
                    "height": window.height,
                    "width": window.width,
                    "transform": src.window_transform(window),
                    "nodata": nodata_value,
                }
            )

            # Temporary filename to store clipped raster
            clipped_fn = os.path.join(output_dir, f"clipped_{os.path.basename(raster)}")
            try:
                with rio.open(clipped_fn, "w", **clipped_meta) as dest:
                    dest.write(clipped_data, 1)
                    # print("Temporary file stored.")
            except Exception as e:
                print(f"Error writing file {clipped_fn}: {e}")
                try:
                    clipped_fn = os.path.join(
                        outputdir2, f"clipped_{os.path.basename(raster)}"
                    )
                    with rio.open(clipped_fn, "w", **clipped_meta) as dest:
                        dest.write(clipped_data, 1)
                        print("Temporary file stored in alterate directory", outputdir2)

                except Exception as f:
                    print(
                        f"Error writing file {clipped_fn} to alternate directory: {f}"
                    )
                    return None
                    # raise

            return clipped_fn  # Return the temporary clipped file name
    except Exception as ex:
        print(f"Error processing file {raster}: {ex}")
        return None


######################################################################


def plot_clipped_rasters(raster1, raster2, bounds=None, title=None):
    """
    Function to plot two clipped rasters side by side with tile bounds overlaid.

    Args:
        raster1 (str): Path to the first raster file.
        raster2 (str): Path to the second raster file.
        bounds (tuple, optional): Bounds of the tile cell (xmin, ymin, xmax, ymax). Defaults to None.
        title (str, optional): Title of the
    """
    # Open the clipped rasters
    with rio.open(raster1) as src1, rio.open(raster2) as src2:
        fig, axes = plt.subplots(1, 2, figsize=(15, 8))

        if title and isinstance(title, str):
            titles = title.split("/")
            if len(titles) != 2:
                raise ValueError(
                    "Title must contain exactly two parts separated by a slash (/)."
                )
        else:
            titles = ["Clipped Raster 1", "Clipped Raster 2"]

        for ax, src, title in zip(axes, [src1, src2], titles):
            # Plot the raster
            extent = [
                src.bounds.left,
                src.bounds.right,
                src.bounds.bottom,
                src.bounds.top,
            ]
            rio.plot.show(src, ax=ax, title=title, cmap="terrain", extent=extent)

            # Overlay tile bounds, if provided
            if bounds:
                min_x, min_y, max_x, max_y = bounds
                ax.plot(
                    [min_x, max_x, max_x, min_x, min_x],
                    [min_y, min_y, max_y, max_y, min_y],
                    color="red",
                    linestyle="--",
                    linewidth=2,
                    label="tile Bounds",
                )
                ax.legend(loc="upper right")
                ax.set_xlabel("Longitude")
                ax.set_ylabel("Latitude")

        plt.tight_layout()
        plt.show()


######################################################################


def find_and_unzip(pathfile):
    """
    This is a function to find a .tif file and/or unzip the .gz file
    if necessary (called from reduce_strip())

    Args:
        pathfile (str): name of the file (as in the df_stats pandas DF)

    Returns:
        dict: Dictionary containing statistics.

    """
    tif_pattern = pathfile + "*_dem.tif"

    # Look for the final .tif file
    tif_files = glob.glob(tif_pattern)
    if tif_files:
        print(f"Found existing .tif file: \n{tif_files[0]}")
        return tif_files[0]

    # If .tif file not found, check for a .gz file and unzip it if necessary
    gzfile_list = glob.glob(pathfile + "*.gz")
    if not gzfile_list:
        print("No .gz file found. Skipping...")
        return None

    gz_file = gzfile_list[0]
    tar_file = gz_file[:-3]  # Remove .gz extension

    # Extract the .gz and .tar files if not done already
    try:
        # Unzip the .gz file to .tar
        with gzip.open(gz_file, "rb") as f_gz_in:
            with open(tar_file, "wb") as f_tar_out:
                shutil.copyfileobj(f_gz_in, f_tar_out)

        # Extract the .tar file to the same directory
        with tarfile.open(tar_file, "r") as tar:
            tar.extractall(path=os.path.dirname(tar_file))
            print("...extracted.")

        os.remove(tar_file)
        # Refresh .tif file search
        tif_files = glob.glob(f"{pathfile}*_dem.tif")
        return tif_files[0] if tif_files else None

    except Exception as e:
        print(f"Error decompressing {gz_file}: {e}")
        if os.path.exists(tar_file):
            os.remove(tar_file)
        return None

######################################################################

def warp_and_calculate_stats(mosaic_clipped_fn, stripdem_clipped_fn):
    """
    Warp rasters to the same resolution, extent, and projection, and calculate statistics.

    Args:
        mosaic_clipped_fn (str): path to the mosaic directory (temporal)
        stripdem_clipped_fn (str): path to the strip directory (temporal)

    Returns:
        dict: Dictionary containing statistics.

    """
    ds_list_clipped = warplib.memwarp_multi_fn(
        [mosaic_clipped_fn, stripdem_clipped_fn],
        extent="intersection",
        res="100",
        t_srs="first",
        r="cubic",
    )
    r1_ds, r2_ds = ds_list_clipped
    r1 = iolib.ds_getma(r1_ds, 1)
    r2 = iolib.ds_getma(r2_ds, 1)
    diff = r2 - r1

    if diff.count() == 0:
        print("No valid overlap between input rasters")
        return None, None, None

    diff_stats = (
        tuple(round(stat, 2) for stat in malib.print_stats(diff))
        if malib.print_stats(diff)
        else None
    )
    rmse = round(float(malib.rmse(diff)), 2)
    mean_r2 = round(float(np.nanmean(r2)), 2)

    return diff_stats, mean_r2, rmse

######################################################################

def reduce_strip(
    strip_name,
    tile_bounds,
    strips_dir,
    mosaic_clipped_fn,
    geocell_i,
    diffndv=-9999.0,
    plotting=False,
    temp_dir=None,
):
    """
    Process a strip DEM to match the spatial extent of a fixed 25000x25000 array,
    apply a bitmask, save the result, and return statistics.

    Args:
        strip_name (str): name of the strip (as in the df_stats pandas DF)
        tile_bounds (tuple): bounds of the super tile (e.g. (-300100.0, -1850100.0, -249900.0, -1799900.0))
        strips_dir (str): path to the strip directory (temporal)
        mosaic_clipped_fn (str): path to the mosaic directory (temporal)
        geocell_i (str): geocell of the strip
        diffndv (float): no-value float (-9999)
        plotting (bool, optional): plots both the mosaic and the strip cropped to the supertile. Defaults to False.
        temp_dir (str): directory to save the processed output array.

    Returns:
        dict: Dictionary containing statistics.
    """
    strippath = f"/media/luna/archive/SATS/OPTICAL/ArcticDEM/ArcticDEM/strips/s2s041/2m/{geocell_i}/SETSM_s2s041_{strip_name}"
    tif_file = find_and_unzip(strippath)
    if not tif_file:
        return None, None, None, None

    try:
        # Handle the bitmask file
        bitmask_pattern = strippath + "*_bitmask.tif"
        bitmask_files = glob.glob(bitmask_pattern)
        if not bitmask_files:
            print("No bitmask .tif file found. Proceeding without mask...")
            bitmask_clipped_fn = None
        else:
            bitmask_clipped_fn = clip_raster_to_cell(
                bitmask_files[0], tile_bounds, strips_dir, nodata_value=6
            )
            if bitmask_clipped_fn is None or not os.path.exists(bitmask_clipped_fn):
                print("Bitmask clipping failed, proceeding without mask...")
                bitmask_clipped_fn = None

        # Clip the strip to the tile bounds
        stripdem_clipped_fn = clip_raster_to_cell(
            tif_file, tile_bounds, strips_dir, diffndv
        )
        if stripdem_clipped_fn is None or not os.path.exists(stripdem_clipped_fn):
            print("Clipping failed, skipping this strip.")
            return None, None, None, None

        # Read the clipped data and its transform
        with rio.open(stripdem_clipped_fn) as src:
            strip_data = src.read(1)
            strip_transform = src.transform
            # strip_nodata = src.nodata

        # Create an empty 25000x25000 array for the tile
        tile_size = (25000, 25000)
        tile_array = np.full(tile_size, diffndv, dtype=np.float32)

        # Get the offsets to place the clipped data into the 25000x25000 array
        x_min, y_max = tile_bounds[0], tile_bounds[3]
        col_start = int((strip_transform.c - x_min) / strip_transform.a)
        row_start = int((y_max - strip_transform.f) / -strip_transform.e)
        col_end = col_start + strip_data.shape[1]
        row_end = row_start + strip_data.shape[0]

        # Place the clipped data into the tile array
        tile_array[row_start:row_end, col_start:col_end] = strip_data

        # Apply the bitmask, if available
        if bitmask_clipped_fn:
            with rio.open(bitmask_clipped_fn) as mask_src:
                bitmask_data = mask_src.read(1)
                tile_array[row_start:row_end, col_start:col_end][
                    bitmask_data != 0
                ] = diffndv

        # Save the full 25000x25000 array
        processed_output_file = os.path.join(temp_dir, f"masked_arr_{strip_name}.npy")
        os.makedirs(os.path.dirname(processed_output_file), exist_ok=True)
        np.save(processed_output_file, tile_array)
        # print(f"processed_output_file saved as {processed_output_file}")

        # Compute stats between mosaic and clipped array
        diff_stats, mean_r2, rmse = warp_and_calculate_stats(
            mosaic_clipped_fn, stripdem_clipped_fn
        )

        if plotting:
            plot_clipped_rasters(
                mosaic_clipped_fn,
                tif_file,
                bounds=tile_bounds,
                title="Mosaic/Original raster",
            )
            plt.close()
            plot_clipped_rasters(
                bitmask_clipped_fn,
                stripdem_clipped_fn,
                bounds=tile_bounds,
                title="Bitmask/Clipped+masked raster",
            )
            plt.close()
            plt.imshow(tile_array, cmap="cool", interpolation="nearest")
            plt.title(f"25000x25000 tile for {strip_name} (Bitmask Applied)")
            plt.colorbar(label="Elevation (m)")
            plt.show()
            plt.close()

        # Cleanup
        os.remove(stripdem_clipped_fn)
        if bitmask_clipped_fn:
            os.remove(bitmask_clipped_fn)
        gc.collect()

        return diff_stats, mean_r2, rmse, processed_output_file

    except Exception as e:
        print(f"Error processing strip {strip_name}: {e}")
        return None, None, None, None


######################################################################


def calculate_statistics(running_sum, running_squared_sum, valid_count):
    """
    Calculate mean and standard deviation from running totals.
    """
    mean_dems = running_sum / valid_count
    sigma = np.sqrt((running_squared_sum / valid_count) - (mean_dems**2))
    mean_dems[valid_count == 0] = np.nan
    sigma[valid_count == 0] = np.nan
    return mean_dems, sigma


######################################################################
def stats_calculator(supertile, subtile, tile_bounds, intersect_dems_df, strip_index_gdf, mosaic_index_gdf, mosaic_dir, strips_dir, stats_columns):
    """
    This function computes the statistics for a tile.

        Parameters:
        supertile (str): The identifier for the supertile.
        subtile (str): The identifier for the subtile.
        tile_bounds (tuple): The geographical bounds of the tile.
        intersect_dems_df (pd.DataFrame): DataFrame containing intersecting DEMs information.
        strip_index_gdf (gpd.GeoDataFrame): GeoDataFrame containing strip index information.
        mosaic_index_gdf (gpd.GeoDataFrame): GeoDataFrame containing mosaic index information.
        mosaic_dir (str): Directory path where mosaic DEMs are stored.
        strips_dir (str): Directory path where strip DEMs are stored.
        stats_columns (list): List of column names for the statistics DataFrame.

        Returns:
        pd.DataFrame: DataFrame containing the computed statistics for the tile.

    """
    tile = f"{supertile}_{subtile}"
    tile_id = tile + "_2m_v4.1"

    # Set file path for tile shapefile and dataframe storage
    df_dir = (
        maindir
        + f"data/ArcticDEM/ArcticDEM_stripfiles/{supertile}/df_csvs/"
    )

    # Ensure the dataframe directory exists
    if not os.path.exists(df_dir):
        os.makedirs(df_dir)

    # Set file path for mosaic DEM
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
    output_stats_file = df_dir + str(tile) + "_stats_df_2m.csv"

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

            except IndexError as e:
                if "list index out of range" in str(e):
                    print("\nFile not downloaded yet... Continue (/) \n")
                    return None
                else:
                    raise

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
    
    # Make sure to delete the temporary files after processing
    if os.path.exists(mosaic_clipped_fn):
        os.remove(mosaic_clipped_fn)

    return df_stats

######################################################################
