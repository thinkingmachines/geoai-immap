import subprocess

def run_cmd(command, check = True):
    try:
        result = subprocess.run(
            command, shell=True, check=check, capture_output=True
        )
    except subprocess.CalledProcessError as exc:
        print(exc.stderr.decode("utf-8"))  # readable gdal error
        raise exc
