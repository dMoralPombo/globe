"""
This module contains the pipeline script for the project.

"""
import gc
import os
# Go to the folder above scripts:
os.chdir("/media/luna/moralpom/globe/github_ready/globe")

# from scripts.functions_all import *
from scripts.functions_config import configuration
from scripts.functions_inter import intersection
from scripts.functions_stats import df_stats_calculator
from scripts.functions_stack import stackador


def main():
    download_only = not bool(input("Download only? (Enter for YES)   "))
    maindir, archdir, res, diffndv, strip_index_gdf, mosaic_index_gdf, stats_columns = configuration()
    supertiles = input("Enter the supertile(s) to process (e.g. '15_38, 16_38'): ").split(",")
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
            tile_id = tile + "_2m_v4.1"

            mosaic_dir = maindir + f"data/ArcticDEM/mosaic/temp/{tile}/"
            strips_dir = maindir + f"data/ArcticDEM/temp2/{tile}/"
            
            tile, tile_coords, tile_bounds, intersect_dems_df = intersection(supertile, subtile, strip_index_gdf, mosaic_index_gdf, archdir)

            if download_only is False:
                df_stats = df_stats_calculator(supertile, subtile, tile_bounds, intersect_dems_df, mosaic_index_gdf, mosaic_dir, strips_dir, stats_columns)
                
                stackador(df_stats, threshold, tile, tile_bounds)
                print("\n\n\nStack run. Proceeding to next tile...\n\n\n")
                gc.collect()
                
    gc.collect()


if __name__ == "__main__":
    main()
