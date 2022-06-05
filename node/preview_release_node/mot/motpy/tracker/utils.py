import importlib.util

# from motpy import Track
from node.preview_release_node.mot.motpy.tracker import Track


def ensure_packages_installed(packages, stop_if_some_missing: bool = True):
    some_missing = False
    for package in packages:
        spec = importlib.util.find_spec(package)
        if spec is None:
            some_missing = True
            print(f'package {package} is not installed')

    if some_missing and stop_if_some_missing:
        print('Please install required python packages to run this script')
        exit(1)


def track_to_string(track: Track) -> str:
    score = track.score if track.score is not None else -1
    return f'ID: {track.id[:8]} | S: {score:.1f} | C: {track.class_id}'
