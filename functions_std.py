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
    tile,
    transform,
    cbar_title="# DEMs",
    cmap="viridis",
    add_grid=True,
    vmin=None,
    vmax=None,
):
    """
    Plot the two main output rasters: number of DEMs used and standard deviation of elevation per pixel.

    Args:
        raster_path (str): Path to the input raster file.
        tile (str): Tile of the raster file.
        cbar_title (str, optional): Title to be displayed in the colorbar. Defaults to '# DEMs'.
        cmap (str, optional): Colour map. Defaults to 'viridis'.
        vmin (float, optional): Minimum value of the colour map. Defaults to None.
        vmax (float, optional): Maximum value of the colour map. Defaults to None.
    """
    title = f"{cbar_title}_{tile}"
    with rio.open(raster_path) as src:
        raster_data = src.read(1)
        raster_data[raster_data == -9999] = np.nan  # Handle no-data values

        # Calculate the extent of the raster in the appropriate CRS
        left, bottom, right, top = src.bounds
        extent = (left, right, bottom, top)

        # Get raster dimensions
        height, width = raster_data.shape

        # Round extent for consistent labeling
        extent_r = [np.round(a, -1) for a in extent]

        # Create transformer from EPSG 3413 to EPSG 4326
        transformer = Transformer.from_crs("EPSG:3413", "EPSG:4326", always_xy=True)

        # Get primary axis ticks (EPSG 3413)
        x_ticks = np.linspace(left, right, 6)[1:-1]  # Skip corners
        y_ticks = np.linspace(bottom, top, 6)[1:-1]  # Skip corners

        # Transform primary axis ticks to secondary system (EPSG 4326)
        lon_ticks, _ = transformer.transform(x_ticks, [bottom] * len(x_ticks))
        _, lat_ticks = transformer.transform([left] * len(y_ticks), y_ticks[::-1])

        # Plotting the raster data
        fig, ax = plt.subplots(figsize=(8, 6))
        img = ax.imshow(
            raster_data, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent, origin="upper"
        )

        # Set primary axis labels (EPSG 3413)
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.set_xlabel("X (m) - EPSG 3413")
        ax.set_ylabel("Y (m) - EPSG 3413")
        ax.set_title(title, fontweight="bold")

        # Add secondary axes (EPSG 4326) with synchronized labels
        secax_x = ax.secondary_xaxis("top")
        secax_x.set_xticks(x_ticks)
        secax_x.set_xticklabels([f"{lon:.1f}" for lon in lon_ticks])
        secax_x.set_xlabel("Longitude (°) - EPSG 4326", labelpad=12)

        secax_y = ax.secondary_yaxis("right")
        secax_y.set_yticks(y_ticks)
        secax_y.set_yticklabels([f"{lat:.1f}" for lat in lat_ticks])
        secax_y.set_ylabel("Latitude (°) - EPSG 4326", labelpad=10, rotation=270)

        # Add grid if requested
        if add_grid:
            ax.grid(
                visible=True, which="both", color="gray", linestyle="--", linewidth=0.5
            )

        # Add colorbar
        cbar = fig.colorbar(img, ax=ax, orientation="vertical", pad=0.15)
        cbar.set_label(cbar_title, labelpad=12, rotation=270)

        plt.tight_layout()

        # Create output directories if they don't exist
        image_dir = os.path.join(
            maindir,
            f"data/ArcticDEM/ArcticDEM_stripfiles/{tile[:5]}/images",
        )

        if not os.path.exists(image_dir):
            os.makedirs(image_dir)

        # Save the PNG image
        png_save_path = os.path.join(image_dir, f"{title}_sec.png")
        plt.savefig(png_save_path, dpi=500)
        print(f"PNG image saved at {png_save_path}")
        plt.show()

        # Save the raster data as a TIFF file
        # tiff_dir = os.path.join(
        #     maindir,
        #     f"data/ArcticDEM/ArcticDEM_stripfiles/{tile[:5]}/arrays",
        # )
        # if not os.path.exists(tiff_dir):
        #     os.makedirs(tiff_dir)

        # tiff_save_path = os.path.join(tiff_dir, f"{title}.tif")

        # Save raster array as TIFF using rasterio
        # NO: already saved by save_raster() using the transform
        # (is there any difference?)
        # with rio.open(
        #     tiff_save_path,
        #     "w",
        #     driver="GTiff",
        #     height=raster_data.shape[0],
        #     width=raster_data.shape[1],
        #     count=1,
        #     dtype=raster_data.dtype,
        #     crs=src.crs,
        #     transform=src.transform,
        # ) as dst:
        #     dst.write(raster_data, 1)


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


# Define reusable utility functions
def initialise_running_totals(cellshape=(25000, 25000), dtype=np.float64):
    """
    Initialize arrays for cumulative calculations.
    """
    running_sum = np.zeros(cellshape, dtype=dtype)
    running_squared_sum = np.zeros(cellshape, dtype=dtype)
    valid_count = np.zeros(cellshape, dtype=dtype)
    print("Arrays initialised.")
    return running_sum, running_squared_sum, valid_count


######################################################################


def process_array(npy_file, running_sum, running_squared_sum, valid_count):
    """
    Load a numpy array, update running totals, and clear memory.
    """
    loaded_array = np.load(npy_file)
    # print(f'Shape of the loaded_array is', loaded_array.shape)
    if loaded_array.shape != running_sum.shape:
        print(f"npy_file {npy_file} has a weird shape")
        return

    # Replace -9999 with NaN
    loaded_array = np.where(loaded_array == -9999, np.nan, loaded_array)

    # Create a mask for valid (non-NaN) values
    maskarr = ~np.isnan(loaded_array)

    # Update running totals using the valid mask
    running_sum[maskarr] += loaded_array[maskarr]
    running_squared_sum[maskarr] += loaded_array[maskarr] ** 2
    valid_count[maskarr] += 1

    # Clean up
    maskarr = []
    loaded_array = []
    del loaded_array, maskarr
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


#########################################################


def stack(
    stackarrays,
    tile,
    tile_bounds,
    plot_every_n,  # Plot one out of every N DEMs
    transform=None,
):
    """
    This is a function to handle the processing of the last part of the pipeline,
    about creating the stdev (and nDEMs) maps.

    Args:
        stackarrays (list): list including the (temporary) arrays stored as npy files
        tile (str): id of the tile.
        tile_bounds (str):
        transform (_type_,): _description_. Defaults to None.
        plot_every_n (int, optional): frequency of plotting. Defaults to 10.
    """
    # Get the tile bounds (x0, y0, x1, y1)
    x0, y0, x1, y1 = tile_bounds

    # Generate the extent (coordinates for the axes)
    extent = [x0, y0, x1, y1]

    # Pre-calculate affine transformation
    transform = Affine(2.0, 0.0, x0, 0.0, -2.0, y1)

    cellshape = (25000, 25000)

    if not stackarrays:
        print("Array is empty")
        return

    # Paths for saving outputs
    resultsdir = os.path.join(
        maindir,
        f"data/ArcticDEM/ArcticDEM_stripfiles/{tile[:5]}/arrays/",
    )
    os.makedirs(resultsdir, exist_ok=True)

    ndems_raster_path = os.path.join(resultsdir, f"{tile}_20_ndems.tif")
    meandems_raster_path = os.path.join(resultsdir, f"{tile}_20_mean.tif")
    std_raster_path = os.path.join(resultsdir, f"{tile}_20_stdev.tif")

    # Initialize running totals
    running_sum, running_squared_sum, valid_count = initialise_running_totals(cellshape)

    # Process each saved array
    for idx, npy_file in enumerate(tqdm(stackarrays, desc="Stacking DEM arrays")):
        try:
            # Update statistics
            print(f"Processing array {idx}")
            process_array_new(npy_file, running_sum, running_squared_sum, valid_count)

            # Plot boundaries for every Nth DEM
            if isinstance(plot_every_n, int) and idx % plot_every_n == 0:
                # Mask valid data (non-NaN pixels)
                data_mask = valid_count > 0
                print("data_mask created")

                # Plot the boundary
                fig, ax = plt.subplots(figsize=(8, 6))
                img = ax.imshow(
                    valid_count, extent=extent, cmap="viridis", origin="upper"
                )

                plt.title(f"DEM Data coverage after {idx+1} strips.")
                plt.suptitle(f"{tile} (with coords)")
                plt.xlabel("Longitude (m) - EPSG 3413")
                plt.ylabel("Latitude (m)")
                ax.grid(True, which="both", linestyle="--", linewidth=0.5)
                plt.colorbar(img)
                plt.show()
                plt.close(fig)

                # Clean up the loaded array
                # loaded_array = []
                data_mask = []
                del data_mask
                gc.collect()

            print(f"DEM (#{idx}) processed")

        except MemoryError:
            print(f"MemoryError encountered while stacking {npy_file}. Skipping...")
        except Exception as e:
            print(f"Error encountered while stacking {npy_file}: {e}. Skipping...")

        gc.collect()

    # Calculate statistics
    mean_dems, sigma = calculate_statistics(
        running_sum, running_squared_sum, valid_count
    )

    # Save results as GeoTIFFs
    save_raster(mean_dems, meandems_raster_path, transform)
    save_raster(valid_count, ndems_raster_path, transform)
    save_raster(sigma, std_raster_path, transform)
    print("mean, ndems and sigma rasters saved.")

    # Plot the rasters
    plot_final_raster(
        ndems_raster_path, tile, transform, "ndems", "viridis", add_grid=True
    )
    plt.close()

    plot_final_raster(
        meandems_raster_path,
        tile,
        transform,
        "meanelev",
        "terrain",
    )
    # plot_final_raster(meandems_raster_path, tile, "meanelev", "terrain")
    plt.close()

    plot_final_raster(
        std_raster_path,
        tile,
        transform,
        "sigma",
        "magma",
        add_grid=True,
        vmin=0,
        vmax=20,
    )
    # plot_final_raster(std_raster_path, tile, "sigma", "magma", vmin=0, vmax=20)
    plt.close()

    plot_final_raster(
        std_raster_path,
        tile,
        transform,
        "sigma",
        "magma",
        add_grid=True,
        vmin=0,
        vmax=50,
    )
    # plot_final_raster(std_raster_path, tile, "sigma_max50", "magma", vmin=0, vmax=50)
    plt.close()

    # Clear arrays from memory
    del stackarrays, running_sum, running_squared_sum, valid_count, mean_dems
    del sigma, extent


def process_array_new(npy_file, running_sum, running_squared_sum, valid_count):
    """
    Load a numpy array, update running totals, and clear memory.
    """
    # Load array and check shape
    loaded_array = np.load(npy_file, mmap_mode="r")
    # Use memory mapping to load large files efficiently
    # print("array loaded with mmap")
    if loaded_array.shape != running_sum.shape:
        print(f"npy_file {npy_file} has a weird shape")
        return

    # Create a mask for valid (non -9999) values
    valid_mask = loaded_array != -9999
    # print("valid_mask")

    # Update running totals using the valid mask
    running_sum[valid_mask] += loaded_array[valid_mask]
    running_squared_sum[valid_mask] += loaded_array[valid_mask] ** 2
    valid_count[valid_mask] += 1
    # print("running_sum, running_squared_sum and valid_count updated")

    valid_mask = []
    loaded_array = []
    del valid_mask, loaded_array
    gc.collect()
    # print("valid_mask and loaded_array deleted, gc collected")


######################################################################


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
        f"\n\n\n\nPART 1: FINDING INTERSECTING DEMS (for tile {tile}\ntile_bounds: {tile_bounds}.\n\n\n\n"
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


def check_intersection(strip_geom, tile_coords):
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
