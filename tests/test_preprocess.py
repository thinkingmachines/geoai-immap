import sys
import os
import pytest

sys.path.insert(0, "../utils")
sys.path.insert(0, "utils")
from general_utils import run_cmd
from env_settings import images_dir, indices_dir

@pytest.mark.skip(reason="focus on predict")
def test_preprocess():
    run_cmd("""
    eval "$(conda shell.bash hook)" &&
    conda activate repo_env &&
    python src/preprocess.py --area='itagui' --start 2021 --end 2021
    """)

    # any year renamed to 2019-2021 to fit original model feature name
    assert os.path.exists(images_dir + "itagui_2019-2020.tif") 
    assert os.path.exists(indices_dir + "indices_itagui_2019-2020.tif") 

