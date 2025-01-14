"""
Functions:
    main(): The main function that runs the pipeline script. It handles user inputs,
    configuration, and processing of supertiles and subtiles.
Modules:
    gc: Garbage collection interface.
    os: Miscellaneous operating system interfaces.
    configuration: Function to set up the configuration for the pipeline.
    intersection: Function to find the intersection of DEMs.
    df_stats_calculator: Function to calculate statistics for the DEMs.
    stackador: Function to stack DEMs based on calculated statistics.
Usage:
    Run this script directly to execute the pipeline. The script will prompt for user
    inputs to control the processing flow.

This module contains the pipeline script for the project.

"""
import gc
import os
# Go to the folder above scripts:
os.chdir("/media/luna/moralpom/globe/github_ready/globe")

# from scripts.functions_all import *
# from scripts.functions_config import configuration
# from scripts.functions_inter import intersection
# from scripts.functions_stats import df_stats_calculator
# from scripts.functions_stack import stackador
from scripts.functions_config import *
from scripts.functions_inter import *
from scripts.functions_stats import *
from scripts.functions_stack import *


def main():
    download_only = not bool(input("Download only? (Enter for YES)   "))
    supertiles = input("Enter the supertile(s) to process (e.g. '15_38, 16_38'): ").split(",")
    maindir, archdir, res, diffndv, strip_index_gdf, mosaic_index_gdf, stats_columns = configuration()
    supertiles = [supertile.strip() for supertile in supertiles]
    subtiles = ["1_1", "1_2", "2_1", "2_2"]
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
            # tile_id = tile + "_2m_v4.1"
            mosaic_dir = maindir + f"data/ArcticDEM/mosaic/temp/{tile}/"
            strips_dir = maindir + f"data/ArcticDEM/temp2/{tile}/"

            # Obtain the coords of the tile and the list of intersecting DEMs
            tile_bounds, intersect_dems_df = intersection(
                supertile,
                subtile,
                strip_index_gdf,
                mosaic_index_gdf,
                archdir
            )

            # Calculate statistics for the DEMs, if download_only is False
            if download_only is False:
                # Calculate statistics for the DEMs
                df_stats = df_stats_calculator(supertile,
                                               subtile,
                                               tile_bounds,
                                               intersect_dems_df,
                                               mosaic_dir,
                                               strips_dir,
                                               stats_columns)

                # Stack the DEMs based on the calculated statistics and obtain final rasters
                stackador(df_stats, threshold, tile, tile_bounds)
                print("\n\n\nStack run. Proceeding to next tile...\n\n\n")
                gc.collect()


if __name__ == "__main__":
    main()
