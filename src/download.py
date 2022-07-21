import sys

sys.path.insert(0, "../utils")
sys.path.insert(0, "utils")
from gee import sen2median
from gee_settings import BBOX, CLOUD_PARAMS
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
    def get_minmaxdt(year_str):
        list_ = year_str.split("-")
        return list_[0] + "-01-01", list_[1] + "-12-31"

    year = f"{start}-{end}"
    cloud_pct, mask = CLOUD_PARAMS[area][year]
    min_dt, max_dt = get_minmaxdt(year)
    sen2median(
        BBOX[area],
        FILENAME=f"gee_{area}_{year}",
        min_dt=min_dt,
        max_dt=max_dt,
        cloud_pct=cloud_pct,
        mask=mask,
        PRODUCT="COPERNICUS/S2",  # L1C
        verbose=1,
        wait=True
    )


if __name__ == "__main__":
    args = parser.parse_args()
    download(args.area, args.start, args.end)
