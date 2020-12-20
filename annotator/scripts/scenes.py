import os
import cv2
import shutil
import numpy as np
import pandas as pd
from pathlib import Path

from tqdm import tqdm

# ------- args ------- #
root = '../dataset/all/'  # todo fix this
# in_filepath = sys.argv[1]
# im_folderpath = sys.argv[2]
# out_crop_folderpath = sys.argv[3]
min_lifetime_noncrossers = 240
min_lifetime_crossers = 60


# ------- args ------- #

# ------- utils ------- #
def textfile_to_array(filename, dtype=float):
    with open(filename) as file:
        data = file.readlines()
        data = [list(map(float, x.split(","))) for x in data]
        data = np.array(data, dtype=dtype)
    return data


# ------- utils ------- #

# ------- main ------- #
# df = pd.read_csv(root+in_filepath)
# # df = df.astype('int')


def scenes(df, root, im_folderpath, out_scene_folderpath, out_csv_filename=None, save=True):
    # prepare folders
    # delete folder if it exists
    if os.path.exists(os.path.join(root, out_scene_folderpath)) and os.path.isdir(
            os.path.join(root, out_scene_folderpath)):
        shutil.rmtree(os.path.join(root, out_scene_folderpath), ignore_errors=True)
    # create all folders
    if not os.path.exists(os.path.join(root, out_scene_folderpath)):
        os.makedirs(os.path.join(root, out_scene_folderpath))

    cap = cv2.VideoCapture(os.path.join(root, im_folderpath))
    video_frame_counter = 0

    new_rows = []
    print("Making scenes")
    pbar = tqdm(total=df["frame"].nunique())

    filepaths = []
    folderpaths = []

    for tt in df["frame"].unique():
        t = tt
        ret = True
        while video_frame_counter <= t:
            # get frame
            ret, im = cap.read()
            if not ret:
                break
            video_frame_counter += 1
        if not ret:
            break
        pbar.update(1)

        cv2.imwrite(os.path.join(root, out_scene_folderpath, str(t).zfill(10) + ".png"), im)
        for row in df[df["frame"] == t].itertuples(index=False, name='Pandas'):
            new_rows.append(row)
            folderpaths = folderpaths + [os.path.join(Path(out_scene_folderpath))]
            filepaths = filepaths + [str(t).zfill(10) + ".png"]

    df_final = pd.DataFrame(new_rows, columns=df.columns)
    df_final['scene_filename'] = filepaths
    df_final['scene_folderpath'] = folderpaths
    if save:
        df_final.to_csv(os.path.join(root, out_csv_filename), index=False)
    return df_final
