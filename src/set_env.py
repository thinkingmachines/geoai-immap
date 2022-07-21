import os
import sys

sys.path.insert(0, "utils")
from env_settings import (
    data_dir,
    adm_dir,
    tmp_dir,
    images_dir,
    indices_dir,
    pos_mask_dir,
    neg_mask_dir,
    model_dir,
    output_dir,
)


def set_env():

    dirs = [
        data_dir,
        adm_dir,
        tmp_dir,
        images_dir,
        indices_dir,
        pos_mask_dir,
        neg_mask_dir,
        model_dir,
        output_dir,
    ]
    for dir_ in dirs:
        if not os.path.exists(dir_):
            os.makedirs(dir_)

    # get area shape file
    if not os.path.exists(f"{adm_dir}admin_bounds.gpkg"):
        raise Exception(
            "Missing admin_bounds.gpkg file. Please download from data.humdata.org > COL Administrative Divisions Shapefiles.zip > adm2 > convert shp to gpkg via QGIS. Or run this command if available: gsutil cp gs://immap-masks/admin_boundaries/admin_bounds.gpkg {adm_dir}"
        )


if __name__ == "__main__":
    set_env()
