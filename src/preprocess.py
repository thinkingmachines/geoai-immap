import geopandas as gpd
from fiona.crs import to_string
from tqdm import tqdm

import sys
import os

sys.path.insert(0, "../utils")
sys.path.insert(0, "utils")
sys.path.insert(0, "../data")
from geoutils import deflatecrop1
from gee_settings import admin2RefN, multipart
from general_utils import run_cmd
import argparse

import os
import operator
from tqdm import tqdm
import pandas as pd
import numpy as np

pd.set_option("use_inf_as_na", True)

import geopandas as gpd
import rasterio as rio

import sys

sys.path.insert(0, "../utils")
import geoutils
from env_settings import (
    adm_dir,
    tmp_dir,
    raw_dir,
    images_dir,
    indices_dir,
    pos_mask_dir,
    neg_mask_dir,
)



parser = argparse.ArgumentParser(
    description="Download preprocess and rollout model on specific area"
)
parser.add_argument(
    "--area",
    type=str,
    default="itagui",
    help="municipality in utils/gee_settings.py",
)
parser.add_argument(
    "--start",
    type=int,
    default=2021,
    help="year to start collecting satellite images for rollout, exact date will be made Jan 1, {year}",
)
parser.add_argument(
    "--end",
    type=int,
    default=2021,
    help="year to end collecting satellite images for rollout, exact date will be made Dec 31, {year}",
)


def preprocess(area, start, end, clear_local=False):
    year = f"{start}-{end}"

    gdf = gpd.read_file(adm_dir + "admin_bounds.gpkg")
    fcrs = to_string({"init": "epsg:4326", "no_defs": True})
    gdf.crs = fcrs

    # create shapefiles for cropping
    area1 = gdf[gdf["admin2RefN"] == admin2RefN[area]]
    area1.to_file(adm_dir + area + ".shp")

    # collect filenames to be processed
    files_ = []

    if area in multipart:
        # just get the largest part
        files_.append(f"gee_{area}_{year}0000000000-0000000000")
    else:
        files_.append(f"gee_{area}_{year}")

    for f in tqdm(files_):
        deflatecrop1(
            raw_filename=f,
            raw_dir=raw_dir,
            output_dir=images_dir,
            adm_dir=adm_dir,
            tmp_dir=tmp_dir,
            bucket="gs://immap-images/test/",
            clear_local=False,
        )

    ##### get filepaths

    area_dict = geoutils.get_filepaths(
        [area], images_dir, indices_dir, pos_mask_dir, neg_mask_dir
    )
    assert area_dict[area]

    ### running per area
    # for area in areas:
    if clear_local:
        run_cmd(
            f"gsutil -q -m cp gs://immap-images/test/{area}*.tif {images_dir}"
        )

    # rename {year} to 2019-2020
    cmd = f"""
    for f in data/images/*_{year}.tif; do mv "$f" "$(echo "$f" | sed s/_{year}/_2019-2020/)"; done
    """
    run_cmd(cmd, check = False)

    area_dict = geoutils.get_filepaths(
        [area], images_dir, indices_dir, pos_mask_dir, neg_mask_dir
    )
    print("Image filepaths:")
    print(area_dict[list(area_dict.keys())[0]])

    area_dict = geoutils.write_indices(area_dict, area, indices_dir, tmp_dir)

    if clear_local:
        run_cmd(
            f"gsutil cp {indices_dir}/indices_*.tif gs://immap-indices/test/"
        )
        run_cmd(f"rm {images_dir}/*.tif")
        run_cmd(f"rm {indices_dir}/indices_*.tif")

    print(f"Done {area}")


if __name__ == "__main__":
    args = parser.parse_args()
    preprocess(args.area, args.start, args.end)
