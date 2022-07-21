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


def download(area, start, end):

    areas = [area]
    years = [f"{start}-{end}"]
    PRODUCT = "COPERNICUS/S2"  # L1C

    data_dir = "../data/"

    adm_dir = data_dir + "admin_bounds/"
    img_dir = data_dir + "images/"
    tmp_dir = data_dir + "tmp/"

    def get_minmaxdt(year_str):
        list_ = year_str.split("-")
        return list_[0] + "-01-01", list_[1] + "-12-31"

    dirs = [adm_dir, img_dir, tmp_dir]
    for dir_ in dirs:
        with pathlib.Path(dir_) as path:
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)

    # get area shape file
    if os.path.exists(f"{adm_dir}admin_bounds.gpkg"):
        run_cmd(
            f"gsutil cp gs://immap-masks/admin_boundaries/admin_bounds.gpkg {adm_dir}"
        )
    gdf = gpd.read_file(adm_dir + "admin_bounds.gpkg")
    fcrs = to_string({"init": "epsg:4326", "no_defs": True})
    gdf.crs = fcrs

    for area in areas:
        for year in years:
            cloud_pct, mask = CLOUD_PARAMS[area][year]
            min_dt, max_dt = get_minmaxdt(year)
            sen2median(
                BBOX[area],
                FILENAME=f"gee_{area}_{year}",
                min_dt=min_dt,
                max_dt=max_dt,
                cloud_pct=cloud_pct,
                mask=mask,
                PRODUCT=PRODUCT,
                verbose=1,
            )


if __name__ == "__main__":
    args = parser.parse_args()
    download(args.area, args.start, args.end)
