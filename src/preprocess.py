import geopandas as gpd
from fiona.crs import to_string
import pathlib
from tqdm import tqdm

import sys
import os

sys.path.insert(0, "../utils")
sys.path.insert(0, "../data")
from gee import sen2median, deflatecrop1
from gee_settings import BBOX, CLOUD_PARAMS, admin2RefN
from geoutils import run_cmd
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

import matplotlib.pyplot as plt


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


def preprocess(area, start, end):
    areas = [area]
    years = [f"{start}-{end}"]
    multipart = ["arauca", "tibu", "bogota", "puertocarreno2"]
    PRODUCT = "COPERNICUS/S2"  # L1C

    data_dir = "../data/"

    adm_dir = data_dir + "admin_bounds/"
    img_dir = data_dir + "images/"
    tmp_dir = data_dir + "tmp/"

    data_dir = "../data/"
    images_dir = data_dir + "images/"
    indices_dir = data_dir + "indices/"
    pos_mask_dir = data_dir + "pos_masks/"
    neg_mask_dir = data_dir + "neg_masks/"

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    if not os.path.exists(indices_dir):
        os.makedirs(indices_dir)
    if not os.path.exists(pos_mask_dir):
        os.makedirs(pos_mask_dir)
    if not os.path.exists(neg_mask_dir):
        os.makedirs(neg_mask_dir)

    # create shapefiles for cropping
    for area in areas:
        area1 = gdf[gdf["admin2RefN"] == admin2RefN[area]]
        area1.to_file(adm_dir + area + ".shp")

    # collect filenames to be processed
    files_ = []

    for area in areas:
        for year in years:
            if area in multipart:
                # just get the largest part
                files_.append(f"gee_{area}_{year}0000000000-0000000000")
            else:
                files_.append(f"gee_{area}_{year}")

    for f in tqdm(files_):
        deflatecrop1(
            raw_filename=f,
            output_dir=img_dir,
            adm_dir=adm_dir,
            tmp_dir=tmp_dir,
            bucket="gs://immap-images/20200613/",
            clear_local=True,
        )

    ##### get filepaths

    area_dict = geoutils.get_filepaths(
        areas, images_dir, indices_dir, pos_mask_dir, neg_mask_dir
    )
    assert area_dict[area]

    ### running per area
    for area in areas:
        run_cmd(
            f"gsutil -q -m cp gs://immap-images/20220309/{area}*.tif {images_dir}"
        )

        # rename 2021-2021 to 2019-2020
        cmd = """
        for f in ../data/images/*_2021-2021.tif; do mv "$f" "$(echo "$f" | sed s/_2021-2021/_2019-2020/)"; done
        """
        run_cmd(cmd)

        area_dict = geoutils.get_filepaths(
            [area], images_dir, indices_dir, pos_mask_dir, neg_mask_dir
        )
        print("Image filepaths:")
        print(area_dict[list(area_dict.keys())[0]])

        area_dict = geoutils.write_indices(area_dict, area, indices_dir)

        run_cmd(
            f"gsutil cp {indices_dir}/indices_*.tif gs://immap-indices/20220309/"
        )
        run_cmd(f"rm {images_dir}/*.tif")
        run_cmd(f"rm {indices_dir}/indices_*.tif")

        print(f"Done {area}")


if __name__ == "__main__":
    args = parser.parse_args()
    preprocess(args.area, args.start, args.end)
