import shutil
import xml.etree.ElementTree as ET
import csv
import pandas as pd
import os
from os.path import join


def jaad_to_pd(dataset_path, in_xml, out_csv='', save=False):
    tree = ET.parse(os.path.join(dataset_path, in_xml))
    root = tree.getroot()
    annotations = [['frame', 'id', 'tlx', 'tly', 'width', 'height', 'walking', 'standing', 'looking', 'incrossing']]
    for track in root.iter('track'):
        if track.attrib['label'] != 'pedestrian':
            continue

        for box in track.iter('box'):
            incrossing = walking = standing = looking = 0
            identifier = None
            frame = box.attrib['frame']
            tlx = float(box.attrib['xtl'])
            tly = float(box.attrib['ytl'])
            _brx = float(box.attrib['xbr'])
            _bry = float(box.attrib['ybr'])
            width = _brx - tlx
            height = _bry - tly
            for attribute in box.iter('attribute'):
                if attribute.attrib['name'] == 'id':
                    identifier = attribute.text
                elif attribute.attrib['name'] == 'cross':
                    incrossing = attribute.text
                    incrossing = 1 if incrossing == 'crossing' else 0
                elif attribute.attrib['name'] == 'action':
                    tex = attribute.text
                    if tex == 'walking':
                        walking = 1
                        standing = 0
                    elif tex == 'standing':
                        standing = 1
                        walking = 0
                elif attribute.attrib['name'] == 'look':
                    looking = 1 if attribute.text == 'looking' else 0
            annotations.append([frame, identifier, tlx, tly, width, height, walking, standing, looking, incrossing])

    if save:
        with open(join(dataset_path, out_csv), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(annotations)
    return pd.DataFrame(annotations[1:], columns=annotations[0], dtype=float)

