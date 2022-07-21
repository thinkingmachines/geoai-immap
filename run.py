import sys

sys.path.insert(0, "../utils")

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

if __name__ == "__main__":
    args = parser.parse_args()
    run_cmd(f"python src/set_env.py")
    run_cmd(
        f"python src/download.py --area={args.area} --start={args.start} --end={args.end}"
    )
    run_cmd(
        f"python src/preprocess.py --area={args.area} --start={args.start} --end={args.end}"
    )
    run_cmd(f"python src/predict.py --area={args.area}")
