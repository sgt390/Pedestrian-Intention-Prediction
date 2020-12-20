import os
import cv2
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tqdm import tqdm

print("Classifying trajectories")

parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str)

args = parser.parse_args()

# ------- args ------- # 
filename = args.filename  # "./ground-truth/Ouchy-1/*.txt"


# ------- args ------- #

# ------- utils ------- #
def get_bbox_position(arr, dtype=float):
    return dtype(arr[0] + arr[2] / 2), dtype(arr[1] + arr[3])


# ------- utils ------- #

# ------- main ------- #
def classify_trajectories(df, save=False):
    # df = pd.read_csv(filename)
    df["lifetime"] = 0
    df["cross"] = 0
    # df = textfile_to_array(in_filepath, float)
    # df = sorted(df, key=lambda x : x[0])
    # df = pd.DataFrame(df, columns=["frame", 'id', 'tlx', 'tly', 'width', 'height', 'score', 'cross', 'incrossing', 'x'])

    print("\nChecking whether each pedestrian crossed the street and computing his lifetime")
    pbar = tqdm(total=df["id"].nunique())
    for i in df["id"].unique():
        pbar.update(1)
        time_cross = 0
        # 1. check if at least 20% of his trajectories is in the crossing  #TODO check this assumptions
        for incrossing in df[df['id'] == i]['incrossing']:
            if incrossing == 1:
                time_cross += 1

        cross = 1 if time_cross > int(df[df['id'] == i]['incrossing'].size*0.20) else 0
        df.loc[df["id"] == i, "cross"] = cross

        # compute his lifetime
        if cross == 0:
            df.loc[df["id"] == i, "lifetime"] = df[df["id"] == i]["frame"].max() - df[df["id"] == i]["frame"].min()
        if cross == 1:
            df.loc[df["id"] == i, "lifetime"] = df[(df["id"] == i) & (df["incrossing"] == 0)]["frame"].max() - \
                                                df[(df["id"] == i) & (df["incrossing"] == 0)]["frame"].min()
    if save:
        df.to_csv(filename[0:-4] + '_trajectory' + ".txt", index=False)
    print("Done assigning labels to pedestrians")
    return df


if __name__ == '__main__':
    df = pd.read_csv(filename)
    classify_trajectories(df, save=True)
