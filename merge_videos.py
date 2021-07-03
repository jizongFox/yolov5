import subprocess
from pathlib import Path

import yaml
from itertools import cycle


def path2Path(path):
    return Path(path) if isinstance(path, str) else path


def load_test_video_names(yaml_path):
    with open(str(yaml_path)) as file:
        return yaml.load(file, Loader=yaml.FullLoader)["test"]


def script_generator(weight, save_dir, source):
    return f" python detect.py   " \
           f"--weights '{weight}'  " \
           f"--augment --project '{save_dir}'  " \
           f"--source '{source}' " \
           f"--name=''"


def main(checkpoint_folder: str, raw_video_folder: str):
    checkpoint_folder = path2Path(checkpoint_folder)
    assert checkpoint_folder.exists()
    assert checkpoint_folder.is_dir()
    raw_video_folder = path2Path(raw_video_folder).absolute()
    assert raw_video_folder.exists()
    assert raw_video_folder.is_dir()
    weight_path = checkpoint_folder / "weights" / "best.pt"
    save_dir = checkpoint_folder / "inference_test"
    test_videos = load_test_video_names(str(checkpoint_folder / "split.yaml"))
    # test_videos = Path("/home/jizong/UserSpace/LabelBox/raw/kurger_test").rglob("*/*.mp4")
    for v in cycle(test_videos):
        source = str(raw_video_folder / str(v))
        script = script_generator(str(weight_path), save_dir=str(save_dir), source=source),
        code = subprocess.run(script, shell=True, executable="/usr/bin/zsh")
        if code.returncode != 0:
            raise RuntimeError(script)


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-folder", required=True, type=str, help="checkpoint folder")
    parser.add_argument("--raw-video-folder", required=True, type=str, help="raw video folder")

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main(args.checkpoint_folder, args.raw_video_folder)
