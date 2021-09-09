import yaml
from gpu_queue import JobSubmitter
from pathlib import Path


def path2Path(path):
    return Path(path) if isinstance(path, str) else path


"""
# !/bin/bash
python - O
detect.py - -weight / home / jizong / UserSpace / 电力项目 / dataset_2 - 3 / checkpoint / weights / best.pt - -source / home / jizong / UserSpace / 电力项目 / dataset2_210616 / videos / 110
kV坪罗线\  # 013.MP4     --name /home/jizong/UserSpace/电力项目/dataset_2-3/checkpoint/detect/videos  --augment &
python - O
detect.py - -weight / home / jizong / UserSpace / 电力项目 / dataset_2 - 3 / checkpoint / weights / best.pt - -source / home / jizong / UserSpace / 电力项目 / dataset2_210616 / videos / 220
kV磁坪一二回线  # 012.MP4     --name /home/jizong/UserSpace/电力项目/dataset_2-3/checkpoint/detect/videos  --augment &
python - O
detect.py - -weight / home / jizong / UserSpace / 电力项目 / dataset_2 - 3 / checkpoint / weights / best.pt - -source / home / jizong / UserSpace / 电力项目 / dataset2_210616 / videos / DJI_0137.MP4 - -name / home / jizong / UserSpace / 电力项目 / dataset_2 - 3 / checkpoint / detect / videos - -augment
wait
"""


def load_test_video_names(yaml_path):
    with open(str(yaml_path)) as file:
        return yaml.load(file, Loader=yaml.FullLoader)["test"]


def script_generator(weight, save_dir, source):
    return f" python detect.py   " \
           f"--weights '{weight}'  " \
           f"--augment " \
           f"--source '{source}' " \
           f"--name={save_dir} "


def find_videos(root):
    video_paths = sorted(Path(root).glob("*.MOV"))
    return [str(x) for x in video_paths]


def main(video_root, save_dir, weight_path):
    videos = find_videos(video_root)
    assert len(videos) > 0, video_root
    jobs = [script_generator(weight=weight_path, save_dir=save_dir, source=v) for v in videos]
    submitor = JobSubmitter(jobs, available_gpus=["0"])
    submitor.submit_jobs()


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-weight", required=True, type=str, help="checkpoint path")
    parser.add_argument("--video-root", "-r", required=True, type=str, help="raw video folder")
    parser.add_argument("--save-dir", "-s", required=True, type=str, help="save-dir")

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main(args.video_root, args.save_dir, args.checkpoint_weight)
