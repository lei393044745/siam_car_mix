# -*- coding:utf-8 -*-
# ! ./usr/bin/env python
# __author__ = 'zzp'

import cv2
import json
import glob
import numpy as np
from os.path import join
from os import listdir

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dir',type=str, default=r'D:\BaiduNetdiskDownload\whispers', help='your vid data dir')
args = parser.parse_args()

whisper_base_path = args.dir
sub_sets = sorted({'train'})

whispers = []
for sub_set in sub_sets:
    sub_set_base_path = join(whisper_base_path, sub_set)
    videos = sorted(listdir(sub_set_base_path))
    s = []
    for vi, video in enumerate(videos):
        print('subset: {} video id: {:04d} / {:04d}'.format(sub_set, vi, len(videos)))
        v = dict()
        v['base_path'] = join(sub_set, video)
        v['HSI'] = join(sub_set, video, 'HSI')
        v['RGB'] = join(sub_set, video, 'RGB')
        v['HSI-FalseColor'] = join(sub_set, video, 'HSI-FalseColor')
        video_base_path = join(sub_set_base_path, video)
        gts_HSI_path = join(video_base_path, 'HSI', 'groundtruth_rect.txt')
        gts_HSI_FalseColor_path = join(video_base_path, 'HSI-FalseColor', 'groundtruth_rect.txt')
        gts_RGB_path = join(video_base_path, 'RGB', 'groundtruth_rect.txt')
        # gts_file = open(gts_path, 'r')
        # gts = gts_file.readlines()
        gts_RGB = np.loadtxt(open(gts_RGB_path, "rb"))

        # get image size
        im_RGB_path = join(video_base_path, 'RGB','0001.jpg')
        im = cv2.imread(im_RGB_path)
        size = im.shape  # height, width
        frame_sz = [size[1], size[0]]  # width,height

        # get all im name
        jpgs = sorted(glob.glob(join(video_base_path, 'RGB', '*.jpg')))

        f = dict()
        v['frame'] = []
        for idx, img_path in enumerate(jpgs):
            f['frame_sz'] = frame_sz
            f['img_path'] = img_path

            gt = gts_RGB[idx]
            bbox = [int(g) for g in gt]   # (x,y,w,h)
            f['bbox'] = bbox
            v['frame'].append(f.copy())
        s.append(v)
    whispers.append(s)
print('save json (raw got10k info), please wait 1 min~')
json.dump(whispers, open('whisper.json', 'w'), indent=4, sort_keys=True)
print('got10k.json has been saved in ./')
