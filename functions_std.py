"""
Functions to be used in the calculation of std (e.g. calc_advanced.py)

@dmoralpombo (based in Jade Bowling's work)
"""

import os
import sys
import geopandas as gpd
from shapely.geometry import shape, mapping, Polygon, MultiPolygon, box
import numpy as np
from pygeotools.lib import timelib, iolib, malib, warplib, geolib
import seaborn as sns
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from osgeo import gdal
import rasterio as rio
from rasterio.mask import mask
from rasterio.plot import plotting_extent, show
from rasterio.transform import from_origin
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pyproj import Transformer, CRS
import fiona
from concurrent.futures import ThreadPoolExecutor
import subprocess
import requests
import tarfile
import gzip
import shutil
import glob
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pdemtools as pdem
import rioxarray as rxr
from affine import Affine
import gc
from mpl_toolkits.axes_grid1 import make_axes_locatable
from natsort import natsorted  # Natural sorting for filenames
import contextily as cx
from geodatasets import get_path
from matplotlib.patches import Rectangle

# Define main directory
maindir = str("/media/luna/moralpom/globe/")
# and mosaic's directory
mosaic_dem = maindir + "data/ArcticDEM/mosaic/arcticdem_mosaic_100m_v4.1_dem.tif"
# Define spatial resolution of the strips
res = 2


def plot_final_raster(
    raster_path,
    region,
    title,
    cbar_title="# DEMs",
    cmap="viridis",
    vmin=None,
    vmax=None,
):
    """This is a function to plot the two main output rasters: number of DEMs used and standard deviation of elevation per per pixel

    Args:
        raster_path (str): path to the input raster file.
        region (str): _description_
        title (str): title to be displayed in the plot
        cbar_title (str, optional): title to be displayed in the colorbar. Defaults to '# DEMs'.
        cmap (str, optional): colour map. Defaults to 'viridis'.
        vmin (float, optional): minimum value of the colour map. Defaults to None.
        vmax (float, optional): maximum value of the colour map. Defaults to None.
    """
    with rio.open(raster_path) as src:
        raster_data = src.read(1)
        raster_data[raster_data == -9999] = np.nan  # Handle no-data values

        # Calculate the extent of the raster in the appropriate CRS
        left, bottom, right, top = src.bounds
        extent = (left, right, bottom, top)

        # Plotting the raster data
        fig, ax = plt.subplots(figsize=(10, 10))
        img = ax.imshow(raster_data, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent)

        # Create a divider for the existing axes instance
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        # Create the colorbar
        cbar = plt.colorbar(img, cax=cax, orientation="vertical")
        cbar.set_label(cbar_title, fontsize=12)

        ax.set_title(title, fontsize=14)
        ax.set_xlabel("X Coordinate", fontsize=12)
        ax.set_ylabel("Y Coordinate", fontsize=12)

        # Create output directories if they don't exist
        image_dir = os.path.join(
            maindir,
            f"data/ArcticDEM/ArcticDEM_stripfiles/ArcticDEM_stripfiles_{region}/images",
        )
        tiff_dir = os.path.join(
            maindir,
            f"data/ArcticDEM/ArcticDEM_stripfiles/ArcticDEM_stripfiles_{region}/arrays",
        )

        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        if not os.path.exists(tiff_dir):
            os.makedirs(tiff_dir)

        # Save the PNG image
        png_save_path = os.path.join(image_dir, f"{title}.png")
        plt.savefig(png_save_path, dpi=500)
        plt.show()

        # Save the raster data as a TIFF file
        tiff_save_path = os.path.join(tiff_dir, f"{title}.tif")

        # Save raster array as TIFF using rasterio
        with rio.open(
            tiff_save_path,
            "w",
            driver="GTiff",
            height=raster_data.shape[0],
            width=raster_data.shape[1],
            count=1,
            dtype=raster_data.dtype,
            crs=src.crs,
            transform=src.transform,
        ) as dst:
            dst.write(raster_data, 1)


def clip_raster_to_cell(raster, bounds, output_dir, nodata_value=-9999):
    """
    This is a function to clip (crop) a raster to a grid cell (defined by its bounds)

    Args:
        raster (str): Path to the input raster file.
        bounds (tuple): Bounds of the grid cell to clip the raster to (xmin, ymin, xmax, ymax).
        output_dir (str): Directory to store the output clipped raster.
        nodata_value (int, optional): Value to be ignored as "no data". Defaults to -9999.
    Returns:
        clipped_fn (str): Path to the clipped raster file, or None if no overlap.
    """
    outputdir2 = "/media/luna/moralpom/globe/data/ArcticDEM/temp/"
    temp_dir = "/media/luna/moralpom/globe/data/ArcticDEM/temp/"

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

    try:
        # Open the raster file
        with rio.open(raster) as src:
            # Check if bounds overlap with raster's extent
            src_bounds = src.bounds
            print(f"Raster bounds: {src_bounds}")
            print(f"Clipping bounds: {bounds}")

            # Check for overlap
            if (
                bounds[2] <= src_bounds.left
                or bounds[0] >= src_bounds.right
                or bounds[3] <= src_bounds.bottom
                or bounds[1] >= src_bounds.top
            ):
                print("No overlap between raster and bounds.")
                return None  # No overlap, so exit the function

            # Create a window using the grid bounds
            window = rio.windows.from_bounds(*bounds, transform=src.transform)

            # Adjust window if it exceeds raster dimensions
            window = window.intersection(
                rio.windows.Window(
                    col_off=0, row_off=0, width=src.width, height=src.height
                )
            )

            print(f"Window to be used for clipping raster {raster}: {window}")

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


# def plot_clipped_rasters(raster1, raster2, bounds=None):
#     """This is a function to plot two clipped rasters
#     with optional grid bounds overlaid

#     Args:
#         raster1 (_type_): _description_
#         raster2 (_type_): _description_
#         bounds (tuple, optional): Bounds of the grid cell (xmin, ymin, xmax, ymax). Defaults to None.
#     """
#     # Open the clipped rasters
#     with rio.open(raster1) as src1, rio.open(raster2) as src2:
#         fig, ax = plt.subplots(1, 2, figsize=(15, 8))

#         # Plot the first raster
#         rio.plot.show(src1, ax=ax[0], title="Clipped Raster 1", cmap="terrain")

#         # If bounds are provided, overlay them
#         if bounds:
#             extent = [bounds[0], bounds[1], bounds[2], bounds[3]]

#             ax[0].add_patch(
#                 Rectangle(
#                     (bounds[0], bounds[1]),  # Lower-left corner
#                     bounds[2] - bounds[0],  # Width
#                     bounds[3] - bounds[1],  # Height
#                     edgecolor="red",
#                     fill=False,
#                     linewidth=2,
#                     label="Grid Bounds",
#                 )
#             )
#             ax[0].legend()

#             min_x, min_y, max_x, max_y = bounds

#         # Plot the second raster
#         rio.plot.show(src2, ax=ax[1], title="Clipped Raster 2", cmap="terrain", extent=extent)

#         # If bounds are provided, overlay them
#         if bounds:
#             ax[1].add_patch(
#                 Rectangle(
#                     (bounds[0], bounds[1]),  # Lower-left corner
#                     bounds[2] - bounds[0],  # Width
#                     bounds[3] - bounds[1],  # Height
#                     edgecolor="red",
#                     fill=False,
#                     linewidth=2,
#                     label="Grid Bounds",
#                 )
#             )
#             ax[1].legend()

#         # Optional: Adjust layout for better visualization
#         plt.tight_layout()
#         plt.show()

#         plt.close()


def plot_clipped_rasters(raster1, raster2, bounds=None, title=None):
    """
    Function to plot two clipped rasters side by side with grid bounds overlaid.

    Args:
        raster1 (str): Path to the first raster file.
        raster2 (str): Path to the second raster file.
        bounds (tuple, optional): Bounds of the grid cell (xmin, ymin, xmax, ymax). Defaults to None.
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

            # Overlay grid bounds, if provided
            if bounds:
                min_x, min_y, max_x, max_y = bounds
                ax.plot(
                    [min_x, max_x, max_x, min_x, min_x],
                    [min_y, min_y, max_y, max_y, min_y],
                    color="red",
                    linestyle="--",
                    linewidth=2,
                    label="Grid Bounds",
                )
                ax.legend(loc="upper right")
                ax.set_xlabel("Longitude")
                ax.set_ylabel("Latitude")

        plt.tight_layout()
        plt.show()


######################################################################


def find_and_unzip(strippath):
    """
    This is a function to find a .tif file and/or unzip the .gz file
    if necessary (called from reduce_strip())

    Args:
        strip_name (str): name of the strip (as in the df_stats pandas DF)
        diffndv (float): no-value float (-9999)

    Returns:
        dict: Dictionary containing statistics.

    """
    tif_pattern = strippath + "*_dem.tif"

    # Look for the final .tif file
    tif_files = glob.glob(tif_pattern)
    if tif_files:
        print(f"Found existing .tif file: \n{tif_files[0]}")
        return tif_files[0]

    # If .tif file not found, check for a .gz file and unzip it if necessary
    gzfile_list = glob.glob(strippath + "*.gz")
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
        tif_files = glob.glob(f"{strippath}*_dem.tif")
        return tif_files[0] if tif_files else None

    except Exception as e:
        print(f"Error decompressing {gz_file}: {e}")
        if os.path.exists(tar_file):
            os.remove(tar_file)
        return None


######################################################################


def apply_bitmask_to_dem(stripdem_clipped_fn, bitmask_clipped_fn, diffndv=-9999):
    """
    This is a function to process two rasters by clipping them to specified bounds, computing differences,
    and calculating statistics.

    Args:
        stripdem_clipped_fn (str): path to the strip directory (temporal)
        mosaic_clipped_fn (str): path to the mosaic directory (temporal)
        diffndv (float, optional): no-value float (by default: -9999)

    Returns:
        dict: Dictionary containing statistics.

    """
    if not bitmask_clipped_fn:
        return

    with rio.open(bitmask_clipped_fn) as mask_ds:
        mask_data = mask_ds.read(1)

    with rio.open(stripdem_clipped_fn) as dem_ds:
        dem_data = dem_ds.read(1)

    # Combine masks
    dem_mask = dem_data == diffndv
    combined_mask = dem_mask | (mask_data != 0)
    masked_dem = np.ma.array(dem_data, mask=combined_mask)

    with rio.open(
        stripdem_clipped_fn,
        "w",
        driver="GTiff",
        height=masked_dem.shape[0],
        width=masked_dem.shape[1],
        count=1,
        dtype=masked_dem.dtype,
        crs=dem_ds.crs,
        transform=dem_ds.transform,
        nodata=diffndv,
    ) as dem_ds_masked:
        dem_ds_masked.write(masked_dem.filled(diffndv), 1)

    print(f"Masked DEM stored in: {stripdem_clipped_fn}")


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


def reduce_strip(
    strip_name,
    grid_bounds,
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
        grid_bounds (tuple): bounds of the super tile (e.g. (-300100.0, -1850100.0, -249900.0, -1799900.0))
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
                bitmask_files[0], grid_bounds, strips_dir, nodata_value=6
            )
            if bitmask_clipped_fn is None or not os.path.exists(bitmask_clipped_fn):
                print("Bitmask clipping failed, proceeding without mask...")
                bitmask_clipped_fn = None

        # Clip the strip to the grid bounds
        stripdem_clipped_fn = clip_raster_to_cell(
            tif_file, grid_bounds, strips_dir, diffndv
        )
        if stripdem_clipped_fn is None or not os.path.exists(stripdem_clipped_fn):
            print("Clipping failed, skipping this strip.")
            return None, None, None, None

        # Read the clipped data and its transform
        with rio.open(stripdem_clipped_fn) as src:
            strip_data = src.read(1)
            strip_transform = src.transform
            strip_nodata = src.nodata

        # Create an empty 25000x25000 array for the grid
        grid_size = (25000, 25000)
        grid_array = np.full(grid_size, diffndv, dtype=np.float32)

        # Get the offsets to place the clipped data into the 25000x25000 array
        x_min, y_max = grid_bounds[0], grid_bounds[3]
        col_start = int((strip_transform.c - x_min) / strip_transform.a)
        row_start = int((y_max - strip_transform.f) / -strip_transform.e)
        col_end = col_start + strip_data.shape[1]
        col_end = row_start + strip_data.shape[0]

        # Place the clipped data into the grid array
        grid_array[row_start:row_end, col_start:col_end] = strip_data

        # Apply the bitmask, if available
        if bitmask_clipped_fn:
            with rio.open(bitmask_clipped_fn) as mask_src:
                bitmask_data = mask_src.read(1)
                grid_array[row_start:row_end, col_start:col_end][
                    bitmask_data != 0
                ] = diffndv

        # Save the full 25000x25000 array
        processed_output_file = os.path.join(temp_dir, f"masked_arr_{strip_name}.npy")
        os.makedirs(os.path.dirname(processed_output_file), exist_ok=True)
        np.save(processed_output_file, grid_array)
        print(f"processed_output_file saved as {processed_output_file}")

        # Compute stats between mosaic and clipped array
        diff_stats, mean_r2, rmse = warp_and_calculate_stats(
            mosaic_clipped_fn, stripdem_clipped_fn
        )

        if plotting:
            plot_clipped_rasters(
                mosaic_clipped_fn,
                tif_file,
                bounds=grid_bounds,
                title="Mosaic/Original raster",
            )
            plt.close()
            plot_clipped_rasters(
                bitmask_clipped_fn,
                stripdem_clipped_fn,
                bounds=grid_bounds,
                title="Bitmask/Clipped+masked raster",
            )
            plt.close()
            plt.imshow(grid_array, cmap="cool", interpolation="nearest")
            plt.title(f"25000x25000 Grid for {strip_name} (Bitmask Applied)")
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


###################################################################


def reduce_strip_old(
    strip_name,
    grid_bounds,
    strips_dir,
    mosaic_clipped_fn,
    geocell_i,
    diffndv=-9999.0,
    plotting=False,
    temp_dir=None,
):
    """
    This is a function to process two rasters by clipping them to specified bounds, computing differences,
    and calculating statistics.

    Args:
        strip_name (str): name of the strip (as in the df_stats pandas DF)
        grid_bounds (tuple): bounds of the super tile (e.g. (-300100.0, -1850100.0, -249900.0, -1799900.0))
        strips_dir (str): path to the strip directory (temporal)
        diffndv (float): no-value float (-9999)
        mosaic_clipped_fn (str): path to the mosaic directory (temporal)
        geocell_i (str): geocell of the strip
        plotting (bool, optional): plots both the mosaic and the strip cropped to the supertile. Defaults to False.

    Returns:
        dict: Dictionary containing statistics.

    """
    strippath = f"/media/luna/archive/SATS/OPTICAL/ArcticDEM/ArcticDEM/strips/s2s041/2m/{geocell_i}/SETSM_s2s041_{strip_name}"
    tif_file = find_and_unzip(strippath)
    if not tif_file:
        return None, None, None, None

    bitmask_pattern = strippath + "*_bitmask.tif"
    # Look for bitmask file
    bitmask_files = glob.glob(bitmask_pattern)
    if not bitmask_files:
        print("No bitmask .tif file found. Proceeding without mask...")
        bitmask_clipped_fn = None
    else:
        bitmask_clipped_fn = clip_raster_to_cell(
            bitmask_files[0], grid_bounds, strips_dir, nodata_value=6
        )
        if bitmask_clipped_fn is None or not os.path.exists(bitmask_clipped_fn):
            print("Bitmask clipping failed, proceeding without mask...")
            bitmask_clipped_fn = None

    try:
        stripdem_clipped_fn = clip_raster_to_cell(
            tif_file, grid_bounds, strips_dir, diffndv
        )
        if stripdem_clipped_fn is None or not os.path.exists(stripdem_clipped_fn):
            print("Clipping failed, skipping this strip.")
            return None, None, None, None
    except Exception as e:
        print(f"Error during clipping: {e}")
        return None, None, None, None

    apply_bitmask_to_dem(stripdem_clipped_fn, bitmask_clipped_fn, diffndv)

    diff_stats, mean_r2, rmse = warp_and_calculate_stats(
        mosaic_clipped_fn, stripdem_clipped_fn
    )

    if plotting is True:
        plot_clipped_rasters(
            mosaic_clipped_fn,
            tif_file,
            bounds=grid_bounds,
            title="Mosaic/Original raster",
        )
        plt.close()
        plot_clipped_rasters(
            bitmask_clipped_fn,
            stripdem_clipped_fn,
            bounds=grid_bounds,
            title="Bitmask/Clipped+masked raster",
        )
        plt.close()

    if temp_dir is None:
        raise ValueError("temp_dir must be specified to save the processed output.")

    # Generate a NumPy array and save it
    with rio.open(stripdem_clipped_fn) as src:
        data = src.read(1)
        no_data_value = src.nodatavals[0]
        if no_data_value is not None:
            data = np.ma.masked_equal(data, no_data_value).filled(fill_value=diffndv)

    processed_output_file = os.path.join(temp_dir, f"masked_arr_{strip_name}.npy")
    os.makedirs(
        os.path.dirname(processed_output_file), exist_ok=True
    )  # Ensure parent dirs
    np.save(processed_output_file, data)
    print(f"processed_output_file saved as {processed_output_file}")

    # Cleanup temporary files
    for file_to_remove in [stripdem_clipped_fn, bitmask_clipped_fn]:
        if file_to_remove and os.path.exists(file_to_remove):
            os.remove(file_to_remove)

    gc.collect()

    return diff_stats, mean_r2, rmse, processed_output_file


######################################################################


# Define reusable utility functions
def initialize_running_totals(cellshape=(25000, 25000), dtype=np.float64):
    """
    Initialize arrays for cumulative calculations.
    """
    running_sum = np.zeros(cellshape, dtype=dtype)
    running_squared_sum = np.zeros(cellshape, dtype=dtype)
    valid_count = np.zeros(cellshape, dtype=dtype)
    return running_sum, running_squared_sum, valid_count


######################################################################


def process_array(npy_file, running_sum, running_squared_sum, valid_count):
    """
    Load a numpy array, update running totals, and clear memory.
    """
    loaded_array = np.load(npy_file)
    mask = ~np.isnan(loaded_array)
    running_sum[mask] += loaded_array[mask]
    running_squared_sum[mask] += loaded_array[mask] ** 2
    valid_count[mask] += 1
    del loaded_array, mask
    gc.collect()


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


def save_raster(data, path, transform, nodata_value=-9999.0):
    """
    Save numpy array to a GeoTIFF raster file.
    """
    data[np.isnan(data)] = nodata_value
    with rio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="float32",
        transform=transform,
    ) as dst:
        dst.write(data.astype(np.float32), 1)


######################################################################


def process_raster_files(
    row,
    n,
    archdir,
    grid_extent,
    x0,
    x1,
    y0,
    y1,
    res=2,
    temp_dir="temp_filled_arrays",
    plot_individual=False,
):
    """
    Processes a single raster file, clips it to the grid extent, and prepares a standardized array.

    Parameters:
        file (str): Filename of the raster to process.
        n (int): Index of the file in the DataFrame.
        df_row (pd.Series): Row of the DataFrame containing file metadata.
        archdir (str): Directory path where raster files are stored.
        grid_extent (GeoDataFrame): Extent of the grid to clip against.
        x0, x1, y0, y1 (float): Spatial bounds of the grid.
        res (float): Resolution of the raster.
        temp_dir (str): Directory to save temporary arrays.
        plot_individual (bool): Whether to plot individual arrays for debugging.

    Returns:
        str: Path of the saved numpy array or None if processing fails.
    """
    file = row.filename

    # Extract grid bounds
    min_x, min_y, max_x, max_y = grid_bounds

    # Create a shapefile with the grid polygon
    schema = {"geometry": "Polygon", "properties": {"id": "int"}}
    shapefile_path = os.path.join(temp_dir, "grid_bounds.shp")

    # Create a polygon from the grid bounds and write it to a shapefile
    with fiona.open(
        shapefile_path, "w", driver="ESRI Shapefile", crs="EPSG:4326", schema=schema
    ) as shp:
        # Create a polygon representing the grid bounds
        polygon = box(min_x, min_y, max_x, max_y)  # create a shapely polygon
        shp.write({"geometry": polygon.__geo_interface__, "properties": {"id": 1}})

    try:
        geocell = row.geocell
        stripname = f"SETSM_s2s041_{file}"
        geocell_dir = os.path.join(archdir, geocell)
        matching_files = glob.glob(os.path.join(geocell_dir, f"*{stripname}*_dem.tif"))

        if not matching_files:
            print(f"File not found for {file}. Skipping...")
            return None

        filename_complete = matching_files[0]

        # Open the shapefile and extract the grid polygons
        with fiona.open(shapefile_path, "r") as shapefile:
            shapes = [feature["geometry"] for feature in shapefile]

        # Open raster file
        # with rxr.open_rasterio(filename_complete, masked=True).squeeze() as src:
        # Open raster file
        with rio.open(filename_complete) as src:
            # Read the full raster (original)
            original_array = src.read(1, masked=True)

            # Use rasterio.mask to apply the shapefile mask
            out_array, out_transform = rio.mask.mask(src, shapes, crop=True)

            # Mask invalid values (optional, depending on your data)
            out_array[(out_array < 0) | (out_array >= 3700)] = np.nan

            # Initialize the filled array with NaNs (same shape as the original raster)
            filled_arr = np.full_like(original_array, np.nan, dtype=np.float32)

            # Masked data is inserted into the filled array
            filled_arr[: out_array.shape[1], : out_array.shape[2]] = out_array[0]

            # Convert the masked array to a regular numpy array by replacing masked values with NaNs
            filled_arr = filled_arr.filled(np.nan)

            # Debugging information
            print(f"Masked array shape: {out_array.shape}")
            print(f"Original array shape: {original_array.shape}")
            print(f"Shapefile contains {len(shapes)} polygons")

            # Save filled array to a temporary file
            os.makedirs(temp_dir, exist_ok=True)
            output_file = os.path.join(temp_dir, f"masked_arr_{n}.npy")
            np.save(output_file, filled_arr)

            # window = rio.windows.from_bounds(*grid_bounds, transform=src.transform)
            # clipped_array = src.read(1, window=window, masked=True)
            # clipped_transform = src.window_transform(window)

            # # Mask invalid values
            # clipped_array[(clipped_array < 0) | (clipped_array >= 3700)] = np.nan

            # # Calculate pixel offsets relative to the raster origin
            # delta_x0 = int((grid_bounds[0] - x0) / res)  # Left bound offset
            # delta_y0 = int((grid_bounds[1] - y0) / res)  # Top bound offset

            # # Ensure offsets are non-negative and within bounds
            # delta_x0 = max(0, delta_x0)
            # delta_y0 = max(0, delta_y0)

            # # Calculate valid dimensions for clipping
            # clipped_height, clipped_width = clipped_array.shape
            # max_width = min(25000 - delta_x0, clipped_width)
            # max_height = min(25000 - delta_y0, clipped_height)

            # # Initialize the array and adjust to align with the bottom-left corner
            # filled_arr = np.full((25000, 25000), np.nan, dtype=np.float32)

            # # Adjust delta_y0 for bottom-left alignment
            # filled_arr[
            #     delta_y0 : delta_y0 + max_height,  # Offset from the bottom
            #     delta_x0 : delta_x0 + max_width,  # Offset from the left
            # ] = clipped_array[:max_height, :max_width]

            # # Debugging information
            # print(f"Delta X: {delta_x0}, Delta Y: {delta_y0}")
            # print(f"Filled array shape: {filled_arr.shape}")
            # print(f"Clipped array shape: {clipped_array.shape}")
            # print(f"Grid bounds: {grid_bounds}")
            # print(f"Raster origin: ({x0}, {y0}), Resolution: {res}")

            # # Save filled array to a temporary file
            # os.makedirs(temp_dir, exist_ok=True)
            # output_file = os.path.join(temp_dir, f"filled_arr_{n}.npy")
            # np.save(output_file, filled_arr)

            # Optional: Plot the filled array
            if plot_individual:
                print(
                    f"Plotting the stripDEM (OG vs processesd):\n{filename_complete.split('/')[-1]}"
                )
                fig, axes = plt.subplots(1, 2, figsize=(15, 8))
                titles = ["Original StripDEM", "Processed StripDEM"]

                for ax, data, title in zip(axes, [original_array, filled_arr], titles):
                    # Determine extent for plotting
                    if title == "Original StripDEM":
                        extent = [
                            src.bounds.left,
                            src.bounds.right,
                            src.bounds.bottom,
                            src.bounds.top,
                        ]
                    else:
                        extent = [x0, x1, y0, y1]

                    img = ax.imshow(
                        data[::-1], cmap="terrain", origin="lower", extent=extent
                    )
                    cbar = plt.colorbar(img, ax=ax)
                    cbar.set_label("Elevation (m)", rotation=270, labelpad=15)

                    # Overlay grid bounds
                    min_x, min_y, max_x, max_y = grid_bounds
                    ax.plot(
                        [min_x, max_x, max_x, min_x, min_x],
                        [min_y, min_y, max_y, max_y, min_y],
                        color="red",
                        linestyle="--",
                        label="Grid Bounds",
                    )
                    ax.legend(loc="upper right")
                    ax.set_title(title)
                    ax.set_xlabel("Longitude")
                    ax.set_ylabel("Latitude")

                plt.tight_layout()
                plt.show()

                del original_array

            # Explicitly clear memory
            del filled_arr, shapes

            return output_file

    except Exception as e:
        print(f"Error processing file {file}: {e}")
        return None


#########################################################


def stack(
    spread_arr_20,
    grid_id,
    maindir,
    transform=None,
    plot_every_n=10,  # Plot one out of every N DEMs
):
    """
    This is a function to handle the processing of the last part of the pipeline,
    about creating the stdev (and nDEMs) maps.

    Args:
        spread_arr_20 (_type_): _description_
        grid_id (_type_): _description_
        maindir (_type_): _description_
        transform (_type_, optional): _description_. Defaults to None.
    """
    cellshape = (25000, 25000)

    if not spread_arr_20:
        print("Array is empty")
        return

    print(f"{grid_id}: ---- 20 m dh ----")

    # Paths for saving outputs
    nodems_raster_path = os.path.join(
        maindir,
        f"data/ArcticDEM/ArcticDEM_stripfiles/{tile}/calc_nodems_sigma_threshold/nodems_nofilter/threshold20/{tile}_20_nodems.tif",
    )
    std_raster_path = os.path.join(
        maindir,
        f"data/ArcticDEM/ArcticDEM_stripfiles/{tile}/calc_nodems_sigma_threshold/stdev/threshold20/{tile}_20_stdev.tif",
    )

    # Initialize running totals
    running_sum, running_squared_sum, valid_count = initialize_running_totals(cellshape)

    # Process each saved array
    for idx, npy_file in enumerate(tqdm(spread_arr_20, desc="Processing DEM arrays")):
        # Update statistics
        process_array(npy_file, running_sum, running_squared_sum, valid_count)

        # Plot boundaries for every Nth DEM
        if idx % plot_every_n == 0:
            loaded_array = np.load(npy_file)
            # Mask valid data (non-NaN pixels)
            data_mask = ~np.isnan(loaded_array)

            # Plot the boundary
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(data_mask, cmap="gray", origin="upper")

            plt.title(f"DEM Data coverage (Index {idx})")
            plt.suptitle(f"{npy_file}")
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            ax.grid(True, which="both", linestyle="--", linewidth=0.5)
            plt.show()
            plt.close(fig)  # Close the plot to save memory

            # Clean up the loaded array
            del loaded_array
            gc.collect()

    # Calculate statistics
    mean_dems, sigma = calculate_statistics(
        running_sum, running_squared_sum, valid_count
    )

    # Save results as GeoTIFFs
    save_raster(mean_dems, nodems_raster_path, transform)
    save_raster(sigma, std_raster_path, transform)
    print(f"{grid_id}: ---- Completed writing nodems and stdev rasters ----")

    # # Plot the rasters
    # plot_final_raster(
    #     nodems_raster_path, region, f"ndems_in_gridcell_{grid_id}", "# DEMs", "viridis"
    # )
    # plt.close()

    # plot_final_raster(
    #     std_raster_path,
    #     region,
    #     f"std_in_gridcell_{grid_id}",
    #     "σ elevation (m)",
    #     "magma",
    # )
    # plt.close()

    # Clear arrays from memory
    del spread_arr_20, running_sum, running_squared_sum, valid_count, mean_dems, sigma
    gc.collect()


######################################################################


# def process_grid_intersections(
#     region,
#     tile,
#     maindir,
#     strip_index_gdf,
#     archdir,
# ):
#     """
#     Processes grid intersections for ArcticDEM strips, downloading and unzipping files as needed.

#     Parameters:
#         region (str): Region name for the supertile.
#         supertile_id (str): Identifier for the supertile.
#         maindir (str): Main directory containing data.
#         strip_index_gdf (GeoDataFrame): Geopandas dataframe with ArcticDEM strip information.
#         archdir (str): Archive directory for StripDEM files.

#     Returns:
#         DataFrame: Dataframe containing intersection data.
#     """
#     supertile_id = tile[:5]

#     # Setup paths
#     supertile_path = os.path.join(
#         maindir, f"data/grids/supertiles/{region}/supertile_{region}.shp"
#     )
#     tile_path = os.path.join(
#         maindir, f"data/grids/supertiles/{region}/tile_{region}.shp"
#     )    
#     df_dir = os.path.join(
#         maindir, f"data/ArcticDEM/ArcticDEM_stripfiles/{tile_id}/df_csvs/"
#     )
#     os.makedirs(df_dir, exist_ok=True)

#     intersection_df_path = os.path.join(df_dir, f"{supertile_id}_df.csv")

#     # Load supertile grid
#     tile_gdf = gpd.read_file(supertile_path)
#     gridcrs = tile_gdf.crs

#     # Extract geometry and bounds
#     grid_coords = tile_gdf.iloc[0]["geometry"]
#     grid_bounds = grid_coords.bounds
#     print("Grid bounds:", grid_bounds)

#     if os.path.exists(intersection_df_path):
#         print(f"Loading existing intersection DataFrame from:\n{intersection_df_path}")
#         return pd.read_csv(intersection_df_path), grid_bounds

#     # Initialize variables
#     lines = []
#     n_intersections = 0

#     # Process intersections
#     for _, strip in tqdm(strip_index_gdf.iterrows(), total=strip_index_gdf.shape[0]):
#         strip_geom = strip["geometry"]
#         strip_name = strip["pairname"]
#         geocell = strip["geocell"]
#         url = strip["fileurl"]
#         acqdate1 = strip["acqdate1"]

#         if check_intersection(strip_geom, grid_coords):
#             n_intersections += 1
#             save_name = (
#                 "_".join([strip_name.split("_")[i] for i in [0, 2, 1, 3]]) + ".tif"
#             )
#             lines.append([grid_id, strip_name, acqdate1, save_name, geocell, url])
#             handle_strip_download(maindir, region, geocell, url, archdir)

#     # Create DataFrame and save results
#     intersect_dems_df = pd.DataFrame(
#         lines,
#         columns=["Grid_ID", "Strip_name", "Acq_date", "File_Name", "Geocell", "url"],
#     ).drop_duplicates(subset=["File_Name"])
#     intersect_dems_df.to_csv(intersection_df_path, index=False)

#     return intersect_dems_df, grid_bounds


######################################################################


def process_grid_intersections_new(
    tile,
    maindir,
    strip_index_gdf,
    archdir,
):
    """
    Processes grid intersections for ArcticDEM strips, downloading and unzipping files as needed.

    Parameters:
        tile (str): Identifier for the tile.
        maindir (str): Main directory containing data.
        strip_index_gdf (GeoDataFrame): Geopandas dataframe with ArcticDEM strip information.
        archdir (str): Archive directory for StripDEM files.

    Returns:
        grid_coords: Dataframe containing intersection data.
    """
    supertile_id = tile[:5]

    # Setup paths for the tile shapefile

    df_dir = os.path.join(
        maindir, f"data/ArcticDEM/ArcticDEM_stripfiles/{supertile_id}/df_csvs/"
    )
    os.makedirs(df_dir, exist_ok=True)

    tile_path = os.path.join(f"/media/luna/archive/SATS/OPTICAL/ArcticDEM/ArcticDEM/tiles/{supertile_id}/{tile}_2m_v4.1/index/{tile}_2m_v4.1_index.shp")

    intersection_df_path = os.path.join(df_dir, f"{tile}_df.csv")

    # Download and unzip the tile shapefile
    handle_tile_download(tile)

    # Load tile grid
    tile_gdf = gpd.read_file(tile_path)
    gridcrs = tile_gdf.crs

    # Extract geometry and bounds
    grid_coords = tile_gdf.iloc[0]["geometry"]
    grid_bounds = grid_coords.bounds
    print(f"\n\n\n\nPART 1: TILE LOADED (grid_bounds: {grid_bounds}). Checking intersection...\n\n\n\n")

    if os.path.exists(intersection_df_path):
        print(f"Loading existing intersection DataFrame from:\n {intersection_df_path}")
        return pd.read_csv(intersection_df_path), grid_coords
    # else:
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

        if check_intersection(strip_geom, grid_coords):
            n_intersections += 1
            save_name = (
                "_".join([strip_name.split("_")[i] for i in [0, 2, 1, 3]]) + ".tif"
            )
            lines.append([tile, strip_name, acqdate1, save_name, geocell, url])
            handle_strip_download(maindir, tile, geocell, url, archdir)
    
    # Create DataFrame and save results
    intersect_dems_df = pd.DataFrame(
        lines,
        columns=["Tile", "Strip_name", "Acq_date", "File_Name", "Geocell", "url"],
    ).drop_duplicates(subset=["File_Name"])

    print("\n\n\n\nPART 2: DOWNLOADING STRIPS... \n\n\n\n")
    
    download_strips(intersect_dems_df)

    intersect_dems_df.to_csv(intersection_df_path, index=False)

    return intersect_dems_df, grid_coords


######################################################################


def check_intersection(strip_geom, grid_coords):
    """
    Checks if a strip geometry intersects with the grid coordinates.

    Parameters:
        strip_geom (Polygon or MultiPolygon): Geometry of the strip.
        grid_coords (Polygon): Grid cell geometry.

    Returns:
        bool: True if intersects, False otherwise.
    """
    if isinstance(strip_geom, Polygon):
        return strip_geom.intersects(grid_coords)
    elif isinstance(strip_geom, MultiPolygon):
        return any(poly.intersects(grid_coords) for poly in strip_geom.geoms)
    return False


######################################################################


def handle_strip_download(maindir, tile, geocell, url, archdir):
    """
    Downloads and unzips a StripDEM file if it does not already exist.

    Parameters:
        maindir (str): Main directory containing data.
        tile (str): name for the tile.
        geocell (str): Geocell identifier.
        url (str): URL of the StripDEM file.
        archdir (str): Archive directory for StripDEM files.
    """
    fname = url.split("/")[-1]
    file_path_arc = os.path.join(archdir, geocell, fname)
    # extracted_dir = os.path.join(
    #     maindir,
    #     f"data/ArcticDEM/ArcticDEM_stripfiles/ArcticDEM_stripfiles_{region}",
    #     fname[:-19],
    # )

    extracted_dir = file_path_arc

    if os.path.exists(extracted_dir):
        print(f"Strip already extracted: {extracted_dir}")
        return

    try:
        extracted_tar_path = file_path_arc[:-3]
        with gzip.open(file_path_arc, "rb") as f_gz_in:
            with open(extracted_tar_path, "wb") as f_tar_out:
                shutil.copyfileobj(f_gz_in, f_tar_out)

        with tarfile.open(extracted_tar_path, "r") as tar:
            tar.extractall(extracted_dir)
        os.remove(extracted_tar_path)
        print(f"Strip extracted: {fname}")

    except EOFError:
        print(f"Error: Corrupted or incomplete .gz file: {file_path_arc}")
        # Optionally, delete the corrupted file to allow re-download
        if os.path.exists(file_path_arc):
            os.remove(file_path_arc)
        print(f"Deleted corrupted file: {file_path_arc}")

    except FileNotFoundError:
        print(f"Strip not found: {file_path_arc}. Skipping.")
        



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
    filename_column = "File_Name"
    geocell_column = "Geocell"

    # Create a list of URLs for files that are missing, with subfolders by geocell
    missing_files = []
    for _, row in tqdm(intersect_dems_df.iterrows()):
        geocell_folder = os.path.join(download_folder, row[geocell_column])
        os.makedirs(
            geocell_folder, exist_ok=True
        )  # Create geocell subfolder if it doesn't exist
        filepath = os.path.join(geocell_folder, row[filename_column])
        unzipped_name = row[url_column].split("/")[-1]
        unzfilepath = os.path.join(geocell_folder, unzipped_name)

        # Add to the missing files list if the file doesn't exist
        #if not os.path.exists(filepath) and not os.path.exists(unzfilepath):
        if not os.path.exists(unzfilepath):
            missing_files.append((geocell_folder, row[url_column]))

    print(f'{len(missing_files)} files missing.')
    ################################################################################
    # Use ThreadPoolExecutor to download files concurrently
    max_workers = 10  # Number of concurrent downloads

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
    print("\n\n\n\nPART 3: ALL STRIPS DOWNLOADED. \n\n\n\n")



# Function to download a file using wget
def download_file(geocell_folder, url):
    # Ensure the geocell folder exists before attempting download
    os.makedirs(geocell_folder, exist_ok=True)
    try:
        # Run wget in quiet mode (-q) to reduce verbosity
        subprocess.run(["wget", "-r", "-N", "-nH","-np", "-q", "-A", ".tar.gz", "--cut-dirs=3", \
                        "--no-host-directories", "-nd", "-P", geocell_folder, url], check=True)
        print(f"Downloaded to {geocell_folder}\n from URL {url}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to download: {url}, error: {e}")
