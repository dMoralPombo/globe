"""
This module contains functions for processing and analyzing Digital Elevation Models (DEMs) 
using various geospatial libraries. The main functionalities include clipping rasters, 
plotting rasters, applying bitmasks, warping and calculating statistics, and handling file
downloads and extractions. The module is designed to work with ArcticDEM data and includes
functions for processing tile intersections, calculating statistics, and stacking arrays.

To be called from calc_advanced.py.

Functions:
    plot_final_raster(raster_path, tile, transform, cbar_title, cmap, add_grid, vmin, vmax):
        Plots the main output rasters: number of DEMs used and standard deviation of elevation per pixel.
    initialise_running_totals(cellshape, dtype):
        Initializes arrays for cumulative calculations.
    process_array_new(npy_file, running_sum, running_squared_sum, valid_count):
        Loads a numpy array, updates running totals, and clears memory using memory mapping.
    calculate_statistics(running_sum, running_squared_sum, valid_count):
        Calculates mean and standard deviation from running totals.
    save_raster(data, path, transform, nodata_value):
        Saves a numpy array to a GeoTIFF raster file.
    stack(stackarrays, tile, tile_bounds, plot_every_n):
        Handles the processing of the last part of the pipeline, creating the stdev and nDEMs maps.
    stackador(df_stats, threshold, tile, tile_bounds):
        Filters and stacks arrays based on a threshold and tile information.

@dmoralpombo (based in Jade Bowling's work)
"""
import os
import numpy as np
from tqdm import tqdm # type: ignore
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import rasterio as rio # type: ignore
from rasterio.transform import from_origin # type: ignore
from pyproj import Transformer # type: ignore
import glob
from affine import Affine # type: ignore
import gc

def create_directory_if_not_exists(directory):
    """
    Create a directory if it does not exist.

    Args:
        directory (str): Path to the directory.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

# Define main directory
maindir = str("/media/luna/moralpom/globe/")
# and mosaic's directory
mosaic_dem = maindir + "data/ArcticDEM/mosaic/arcticdem_mosaic_100m_v4.1_dem.tif"
# Define spatial resolution of the strips
res = 2

def stackador(df_stats, threshold, tile, tile_bounds):
    """
    Filters and stacks arrays based on a threshold and tile information.

        df_stats (pd.DataFrame): DataFrame containing statistics with a column "std_dh" for standard deviation of height differences and a column "filename" for core filenames.
        threshold (float): Threshold value for filtering the standard deviation of height differences.
        tile (str): Tile identifier used in directory paths.
        tile_bounds (tuple): Bounds of the tile used in the stacking process.

    Returns:
        None
    """
    # Filter DSMs that are less than threshold = 50 sigma dh (originally it was 20 sigma)
    spread_filtered_df = df_stats[df_stats["std_dh"] < threshold]
    spread_filtered_df = spread_filtered_df.drop_duplicates()
    print(f"Length of spread_filtered_df_20: {len(spread_filtered_df)}")
    temp_dir_filledarrays = f"/media/luna/moralpom/globe/data/ArcticDEM/temp/filled_arrays/{tile}/masked/"

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

######################################################################

def stack(
    stackarrays,
    tile,
    tile_bounds,
    plot_every_n,  # Plot one out of every N DEMs
    #transform=None,
):
    """
    This is a function to handle the processing of the last part of the pipeline,
    about creating the stdev (and nDEMs) maps.

    Args:
        stackarrays (list): list including the (temporary) arrays stored as npy files
        tile (str): id of the tile.
        tile_bounds (tuple): Bounds of the tile in the format (x0, y0, x1, y1).
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
    create_directory_if_not_exists(resultsdir)

    ndems_raster_path = os.path.join(resultsdir, f"{tile}_20_ndems.tif")
    meandems_raster_path = os.path.join(resultsdir, f"{tile}_20_mean.tif")
    std_raster_path = os.path.join(resultsdir, f"{tile}_20_stdev.tif")

    # Initialize running totals
    running_sum, running_squared_sum, valid_count = initialise_running_totals(cellshape)
    
    # Process and plot the DEMs
    def plot_dem(valid_count, idx, extent, tile, plot_every_n):
        """
        Plot the DEMs (every n).

        Args:
            valid_count (np.array): Array containing the count of valid DEM values.
            idx (int): Index of the current DEM being processed.
            extent (list): List containing the coordinates for the axes.
            tile (str): Tile identifier used in directory paths.
            plot_every_n (int): Frequency of plotting. Plots one out of every N DEMs.
        """
        if isinstance(plot_every_n, int) and idx % plot_every_n == 0:
            data_mask = valid_count > 0
            fig, ax = plt.subplots(figsize=(8, 6))
            img = ax.imshow(
                valid_count, extent=extent, cmap="viridis", origin="upper"
            )

            # Additions to the plot
            plt.title(f"DEM Data coverage after {idx+1} strips.")
            plt.suptitle(f"{tile} (with coords)")
            plt.xlabel("Longitude (m) - EPSG 3413")
            plt.ylabel("Latitude (m)")
            ax.grid(True, which="both", linestyle="--", linewidth=0.5)
            plt.colorbar(img)
            plt.show()

            # Clean up the loaded array and delete variable data_mask
            plt.close(fig)
            data_mask = []
            del data_mask
            gc.collect()


    # Process each saved array
    for idx, npy_file in enumerate(tqdm(stackarrays, desc="Stacking DEM arrays")):
        try:
            # Update statistics
            print(f"Processing array {idx}")
            process_array_new(npy_file, running_sum, running_squared_sum, valid_count)

            # Plot the DEMs (every n)
            plot_dem(stackarrays, valid_count, idx, extent, tile, plot_every_n)

            print(f"DEM (#{idx}) processed")

        except MemoryError:
            print(f"MemoryError encountered while stacking {npy_file}. Skipping...")
        except Exception as e:
            print(f"Error encountered while stacking {npy_file}: {e}. Skipping...")

        gc.collect()

    # Define a function to calculate statistics:
    def calculate_statistics(running_sum, running_squared_sum, valid_count):
        """
        Calculate mean and standard deviation from running totals.

        Args:
            running_sum (np.array): Running sum of DEM values.
            running_squared_sum (np.array): Running sum of squared DEM values.
            valid_count (np.array): Running count of valid DEM values.
        
        Returns:
            mean_dems (np.array): Mean elevation values.
            sigma (np.array): Standard deviation of elevation values.
        """
        mean_dems = running_sum / valid_count
        with np.errstate(divide='ignore', invalid='ignore'):
            mean_dems = np.where(valid_count != 0, running_sum / valid_count, np.nan)
        mean_dems[valid_count == 0] = np.nan
        sigma[valid_count == 0] = np.nan
        return mean_dems, sigma

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
        ndems_raster_path, 
        tile, transform, 
        "ndems",
        "viridis", 
        add_grid=True
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

    def plot_quad_subplot(raster1, raster2, raster3, raster4, titles):
        """
        Plot four rasters in a 2x2 grid within a single figure.

        Parameters:
        - raster1, raster2, raster3, raster4: The file paths or arrays for the rasters to be plotted.
        - titles: List of titles for each raster.

        Returns:
        - A Matplotlib figure showing the four rasters in a 2x2 layout.
        """
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        ax_list = axes.flatten()

        rasters = [raster1, raster2, raster3, raster4]
        
        for i, ax in enumerate(ax_list):
            plot_final_raster(
                raster_path=rasters[i],
                ax=ax,
                title=titles[i],
                save_output=False  # Avoid saving directly; modify function to plot on passed `ax`
            )
        
        plt.tight_layout()
        plt.show()
        
        # Create output directories if they don't exist
        image_dir = os.path.join(
            maindir,
            f"data/ArcticDEM/ArcticDEM_stripfiles/{tile[:5]}/images",
        )

        if not os.path.exists(image_dir):
            create_directory_if_not_exists(image_dir)
        # Save the PNG image
        quad_save_path = os.path.join(image_dir, f"tile_quad.png")
        plt.savefig(quad_save_path, dpi=700)


    # Example usage:
    plot_quad_subplot(
        raster1=meandems_raster_path,
        raster2=ndems_raster_path,
        raster3=std_raster_path,
        raster4=std_raster_path,
        titles=["Mean elev.", "# DEMs", "sigma", "sigma"]
    )

    # Clear arrays from memory
    del stackarrays, running_sum, running_squared_sum, valid_count, mean_dems
    del sigma, extent, transform

#########################################################

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


##########################################################################################

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
        transform (Affine): Affine transformation for the raster.
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
            create_directory_if_not_exists(image_dir)
        # Save the PNG image
        png_save_path = os.path.join(image_dir, f"{title}_sec.png")
        plt.savefig(png_save_path, dpi=500)
        print(f"PNG image saved at {png_save_path}")
        plt.show()
