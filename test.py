# split_data.py
import os, random, shutil

SRC = "faces/all"
DST = "faces"

for cls in ["real", "fake"]:
    videos = os.listdir(os.path.join(SRC, cls))
    random.shuffle(videos)

    split = int(0.8 * len(videos))

    for i, v in enumerate(videos):
        subset = "train" if i < split else "val"
        src = os.path.join(SRC, cls, v)
        dst = os.path.join(DST, subset, cls, v)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copytree(src, dst)
