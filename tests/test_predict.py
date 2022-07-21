import sys
import pytest

sys.path.insert(0, "../utils")
sys.path.insert(0, "utils")
from general_utils import run_cmd
from env_settings import output_dir
from glob import glob

def test_predict():
    run_cmd("""
    eval "$(conda shell.bash hook)" &&
    conda activate repo_env &&
    python src/predict.py --area='itagui'
    """)

    probmaps = glob(output_dir + "ensembled/*_itagui.tif")
    assert len(probmaps)>0

