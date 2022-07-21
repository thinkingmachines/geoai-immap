import sys
import os
import pytest

sys.path.insert(0, "../utils")
sys.path.insert(0, "utils")
from general_utils import run_cmd

@pytest.mark.skip(reason="needs GEE authentication")
def test_download():
    run_cmd("""
    eval "$(conda shell.bash hook)" &&
    conda activate ee &&
    python src/download.py --area='itagui' --start 2021 --end 2021
    """)
    run_cmd(
        "gsutil cp gs://immap-gee/gee_itagui_2021-2021.tif downloaded.tif"
    )
    assert os.path.exists("downloaded.tif")
    run_cmd("rm downloaded.tif")
