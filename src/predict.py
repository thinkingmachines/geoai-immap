import os
import joblib
import pandas as pd
import numpy as np
from glob import glob
import argparse

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

import sys

sys.path.insert(0, "../utils")
import model_utils
import geoutils
from geoutils import run_cmd


parser = argparse.ArgumentParser(
    description="Download preprocess and rollout model on specific area"
)
parser.add_argument(
    "--area",
    type=str,
    default="itagui",
    help="municipality in utils/gee_settings.py",
)


SEED = 42


def predict(area):
    areas = [area]
    version = "20200509"
    data_dir = "../data/"
    model_dir = "../models/"
    output_dir = "../outputs/probmaps/"
    input_file = data_dir + "{}_dataset.csv".format(version)

    images_dir = data_dir + "images/"
    indices_dir = data_dir + "indices/"
    pos_mask_dir = data_dir + "pos_masks/"
    neg_mask_dir = data_dir + "neg_masks/"
    tmp_dir = data_dir + "tmp/"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    run_cmd(f"gsutil cp gs://immap-models/* {model_dir}")

    model_names = ["LR_30k", "RF_30k"]
    models = []
    for model_name in model_names:
        filename = "{}model_{}.sav".format(model_dir, model_name)
        models.append(joblib.load(filename))

    features = [
        "B1_2015-2016",
        "B2_2015-2016",
        "B3_2015-2016",
        "B4_2015-2016",
        "B5_2015-2016",
        "B6_2015-2016",
        "B7_2015-2016",
        "B8_2015-2016",
        "B9_2015-2016",
        "B10_2015-2016",
        "B11_2015-2016",
        "B12_2015-2016",
        "ndvi_2015-2016",
        "ndbi_2015-2016",
        "savi_2015-2016",
        "mndwi_2015-2016",
        "ui_2015-2016",
        "nbi_2015-2016",
        "brba_2015-2016",
        "nbai_2015-2016",
        "mbi_2015-2016",
        "baei_2015-2016",
        "B1_2017-2018",
        "B2_2017-2018",
        "B3_2017-2018",
        "B4_2017-2018",
        "B5_2017-2018",
        "B6_2017-2018",
        "B7_2017-2018",
        "B8_2017-2018",
        "B9_2017-2018",
        "B10_2017-2018",
        "B11_2017-2018",
        "B12_2017-2018",
        "ndvi_2017-2018",
        "ndbi_2017-2018",
        "savi_2017-2018",
        "mndwi_2017-2018",
        "ui_2017-2018",
        "nbi_2017-2018",
        "brba_2017-2018",
        "nbai_2017-2018",
        "mbi_2017-2018",
        "baei_2017-2018",
        "B1_2019-2020",
        "B2_2019-2020",
        "B3_2019-2020",
        "B4_2019-2020",
        "B5_2019-2020",
        "B6_2019-2020",
        "B7_2019-2020",
        "B8_2019-2020",
        "B9_2019-2020",
        "B10_2019-2020",
        "B11_2019-2020",
        "B12_2019-2020",
        "ndvi_2019-2020",
        "ndbi_2019-2020",
        "savi_2019-2020",
        "mndwi_2019-2020",
        "ui_2019-2020",
        "nbi_2019-2020",
        "brba_2019-2020",
        "nbai_2019-2020",
        "mbi_2019-2020",
        "baei_2019-2020",
    ]

    for area in areas:
        # download images and indices
        run_cmd(
            f"gsutil -q -m cp gs://immap-images/20220309/{area}_*.tif {images_dir}"
        )
        run_cmd(
            f"gsutil -q -m cp gs://immap-indices/20220309/indices_{area}_*.tif {indices_dir}"
        )
        cmd = """
        for f in ../data/images/*_2020-2021.tif; do mv "$f" "$(echo "$f" | sed s/_2020-2021/_2019-2020/)"; done
        """
        run_cmd(cmd)

        # run prediction on 2 models
        area_dict = geoutils.get_filepaths([area], images_dir, indices_dir)
        for model, model_name in zip(models, model_names):
            out_dir = output_dir + model_name + "/"
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            output = "{}{}_{}_{}.tif".format(out_dir, version, area, model_name)
            print(output)
            # if not os.path.isfile(output):
            geoutils.get_preds_windowing(
                area=area,
                area_dict=area_dict,
                model=model,
                tmp_dir=tmp_dir,
                best_features=features,
                output=output,
                grid_blocks=9,
                threshold=0,
            )

        # run ensembling
        filename1 = "{0:}{3:}/{1:}_{2:}_{3:}.tif".format(
            output_dir, version, area, model_names[0]
        )
        filename2 = "{0:}{3:}/{1:}_{2:}_{3:}.tif".format(
            output_dir, version, area, model_names[1]
        )
        out_dir = output_dir + "ensembled/"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        output_file = "{}{}_{}.tif".format(out_dir, version, area)
        geoutils.get_rasters_merged(
            filename1, filename2, output_file, tmp_dir, grid_blocks=9
        )

        # upload result to output folder
        for model_name in ["ensembled", "LR_30k", "RF_30k"]:
            out_dir = output_dir + model_name + "/"
            files = glob(out_dir + f"*_{area}.tif")
            for filename in files:
                bucket = "gs://immap-output/{}/{}/".format(version, model_name)
                run_cmd(f"gsutil cp {filename} {bucket}")

        # delete images and indices
        run_cmd(f"rm {images_dir}/*.tif")
        run_cmd(f"rm {indices_dir}/indices_*.tif")


if __name__ == "__main__":
    args = parser.parse_args()
    predict(args.area)
