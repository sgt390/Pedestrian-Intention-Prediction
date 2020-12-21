from annotator.scripts.jaad_xml_to_pd import jaad_to_pd
from annotator.scripts.classify_trajectories_jaad import classify_trajectories
from annotator.scripts.hungarian_jaad import hungarian
from annotator.scripts.crop_pedestrians import crop_pedestrians
from annotator.scripts.scenes import scenes
from os.path import join
import os
import shutil

RAW_XML = 'raw_annotations'
RAW_VIDEOS = 'raw'
ANNOTATIONS = 'annotations'
CROPS = 'crops'
SCENES = 'scenes'
ALL_ROOT = join('dataset', 'all')

TRAIN = join('dataset', 'train')
VALIDATION = join('dataset', 'val')
VAL_SPLIT = 0.2

drawtrajectories = False


def generate_groundtruth(in_xml_filename, in_video_name):
    foldername = in_video_name.split('.')[0]
    out_annotated_csv = in_xml_filename.split('.')[0] + '.txt'
    df = jaad_to_pd(ALL_ROOT, join(RAW_XML, in_xml_filename), save=False)
    if len(df['frame']) > 0:
        df = classify_trajectories(df, save=False)
        df = hungarian(df, 50, 60, save=False)
        df = crop_pedestrians(df, ALL_ROOT, join(RAW_VIDEOS, in_video_name), join(CROPS, foldername), join(ANNOTATIONS, out_annotated_csv), save=False)
        if df is not None:
            df = scenes(df, ALL_ROOT, join(RAW_VIDEOS, in_video_name), join(SCENES, foldername),
                    join(ANNOTATIONS, out_annotated_csv), save=True)


def split_dataset():
    files = sorted(os.listdir(join(ALL_ROOT, ANNOTATIONS)))
    crops = sorted(os.listdir(join(ALL_ROOT, CROPS)))
    scenes = sorted(os.listdir(join(ALL_ROOT, SCENES)))
    n_val = int(len(files) * VAL_SPLIT)
    train_files, val_files = files[n_val:], files[:n_val]
    train_crops, val_crops = crops[n_val:], crops[:n_val]
    train_scenes, val_scenes = scenes[n_val:], scenes[:n_val]

    if os.path.exists(TRAIN) and os.path.isdir(TRAIN):
        shutil.rmtree(os.path.join(TRAIN), ignore_errors=True)
    if not os.path.exists(TRAIN):
        os.makedirs(os.path.join(TRAIN, ANNOTATIONS))
        os.makedirs(os.path.join(TRAIN, CROPS))
        os.makedirs(os.path.join(TRAIN, SCENES))

    if os.path.exists(VALIDATION) and os.path.isdir(VALIDATION):
        shutil.rmtree(os.path.join(VALIDATION), ignore_errors=True)
    if not os.path.exists(VALIDATION):
        os.makedirs(os.path.join(VALIDATION, ANNOTATIONS))
        os.makedirs(os.path.join(VALIDATION, CROPS))
        os.makedirs(os.path.join(VALIDATION, SCENES))

    for f in train_files:
        shutil.move(join(ALL_ROOT, ANNOTATIONS, f), join(TRAIN, ANNOTATIONS))
    for f in val_files:
        shutil.move(join(ALL_ROOT, ANNOTATIONS, f), join(VALIDATION, ANNOTATIONS))
    for f in train_crops:
        shutil.move(join(ALL_ROOT, CROPS, f), join(TRAIN, CROPS))
    for f in val_crops:
        shutil.move(join(ALL_ROOT, CROPS, f), join(VALIDATION, CROPS))
    for f in train_scenes:
        shutil.move(join(ALL_ROOT, SCENES, f), join(TRAIN, SCENES))
    for f in val_scenes:
        shutil.move(join(ALL_ROOT, SCENES, f), join(VALIDATION, SCENES))


def remove_folder(foldername):
    if os.path.exists(foldername) and os.path.isdir(foldername):
        shutil.rmtree(foldername, ignore_errors=True)
    # create all folders
    if not os.path.exists(foldername):
        os.makedirs(foldername)


if __name__ == '__main__':
    remove_folder(os.path.join(ALL_ROOT, ANNOTATIONS))
    remove_folder(os.path.join(ALL_ROOT, SCENES))
    remove_folder(os.path.join(ALL_ROOT, CROPS))

    files_xml = sorted(os.listdir(join(ALL_ROOT, RAW_XML)))
    files_video = sorted(os.listdir(join(ALL_ROOT, RAW_VIDEOS)))
    for in_xml, in_video in zip(sorted(files_xml), sorted(files_video)):
        generate_groundtruth(in_xml, in_video)
    split_dataset()
