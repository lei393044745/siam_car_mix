# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import logging
import sys
import os
from collections import namedtuple
Corner = namedtuple('Corner', 'x1 y1 x2 y2')

import cv2
import numpy as np
from torch.utils.data import Dataset

from pysot.utils.bbox import center2corner, Center
from pysot.datasets.augmentation import Augmentation
from pysot.core.config import cfg
from os.path import join

logger = logging.getLogger("global")

# setting opencv
pyv = sys.version[0]
if pyv[0] == '3':
    cv2.ocl.setUseOpenCL(False)


class SubDataset(object):
    def __init__(self, name, root, anno, frame_range, num_use, start_idx):
        cur_path = os.path.dirname(os.path.realpath(__file__))
        self.name = name
        self.root = r'D:\py\HSI\siam_car_mix_data\crop511'
        self.anno = r'D:\py\HSI\siam_car_mix_data\train.json'
        self.frame_range = frame_range
        self.num_use = num_use
        self.start_idx = start_idx
        logger.info("loading " + name)
        with open(self.anno, 'r') as f:
            meta_data = json.load(f)
            meta_data = self._filter_zero(meta_data)

        for video in list(meta_data.keys()):
            for track in meta_data[video]:
                frames = meta_data[video][track]
                frames = list(map(int,
                              filter(lambda x: x.isdigit(), frames.keys())))
                frames.sort()
                meta_data[video][track]['frames'] = frames
                if len(frames) <= 0:
                    logger.warning("{}/{} has no frames".format(video, track))
                    del meta_data[video][track]

        for video in list(meta_data.keys()):
            if len(meta_data[video]) <= 0:
                logger.warning("{} has no tracks".format(video))
                del meta_data[video]

        self.labels = meta_data
        self.num = len(self.labels)
        self.num_use = self.num if self.num_use == -1 else self.num_use
        self.videos = list(meta_data.keys())
        logger.info("{} loaded".format(self.name))
        self.path_format = '{}.{}.{}.jpg'
        self.pick = self.shuffle()

    def _filter_zero(self, meta_data):
        meta_data_new = {}
        for video, tracks in meta_data.items():
            new_tracks = {}
            for trk, frames in tracks.items():
                new_frames = {}
                for frm, bbox in frames.items():
                    if not isinstance(bbox, dict):
                        if len(bbox) == 4:
                            x1, y1, x2, y2 = bbox
                            w, h = x2 - x1, y2 - y1
                        else:
                            w, h = bbox
                        if w <= 0 or h <= 0:
                            continue
                    new_frames[frm] = bbox
                if len(new_frames) > 0:
                    new_tracks[trk] = new_frames
            if len(new_tracks) > 0:
                meta_data_new[video] = new_tracks
        return meta_data_new

    def log(self):
        logger.info("{} start-index {} select [{}/{}] path_format {}".format(
            self.name, self.start_idx, self.num_use,
            self.num, self.path_format))

    def shuffle(self):
        lists = list(range(self.start_idx, self.start_idx + self.num))
        pick = []
        while len(pick) < self.num_use:
            np.random.shuffle(lists)
            pick += lists
        return pick[:self.num_use]

    def get_image_anno(self, video, track, frame):
        frame = "{:06d}".format(frame)
        image_path = os.path.join(self.root, video,
                                  self.path_format.format(frame, track, 'x'))
        image_anno = self.labels[video][track][frame]
        return image_path, image_anno

    def get_positive_pair(self, index):
        video_name = self.videos[index]
        video = self.labels[video_name]
        track = np.random.choice(list(video.keys()))
        track_info = video[track]

        frames = track_info['frames']
        template_frame = np.random.randint(0, len(frames))
        left = max(template_frame - self.frame_range, 0)
        right = min(template_frame + self.frame_range, len(frames)-1) + 1
        search_range = frames[left:right]
        template_frame = frames[template_frame]
        search_frame = np.random.choice(search_range)
        return self.get_image_anno(video_name, track, template_frame), \
            self.get_image_anno(video_name, track, search_frame)

    def get_random_target(self, index=-1):
        if index == -1:
            index = np.random.randint(0, self.num)
        video_name = self.videos[index]
        video = self.labels[video_name]
        track = np.random.choice(list(video.keys()))
        track_info = video[track]
        frames = track_info['frames']
        frame = np.random.choice(frames)
        return self.get_image_anno(video_name, track, frame)

    def __len__(self):
        return self.num


class TrkDataset(Dataset):
    def __init__(self,):
        super(TrkDataset, self).__init__()

        # create sub dataset
        self.all_dataset = []
        start = 0
        self.num = 0
        for name in cfg.DATASET.NAMES:
            subdata_cfg = getattr(cfg.DATASET, name)
            sub_dataset = SubDataset(
                    name,
                    subdata_cfg.ROOT,
                    subdata_cfg.ANNO,
                    subdata_cfg.FRAME_RANGE,
                    subdata_cfg.NUM_USE,
                    start
                )
            start += sub_dataset.num
            self.num += sub_dataset.num_use

            sub_dataset.log()
            self.all_dataset.append(sub_dataset)

        # data augmentation
        self.template_aug = Augmentation(
                cfg.DATASET.TEMPLATE.SHIFT,
                cfg.DATASET.TEMPLATE.SCALE,
                cfg.DATASET.TEMPLATE.BLUR,
                cfg.DATASET.TEMPLATE.FLIP,
                cfg.DATASET.TEMPLATE.COLOR
            )
        self.search_aug = Augmentation(
                cfg.DATASET.SEARCH.SHIFT,
                cfg.DATASET.SEARCH.SCALE,
                cfg.DATASET.SEARCH.BLUR,
                cfg.DATASET.SEARCH.FLIP,
                cfg.DATASET.SEARCH.COLOR
            )
        videos_per_epoch = cfg.DATASET.VIDEOS_PER_EPOCH
        self.num = videos_per_epoch if videos_per_epoch > 0 else self.num
        self.num *= cfg.TRAIN.EPOCH
        self.pick = self.shuffle()

    def shuffle(self):
        pick = []
        m = 0
        while m < self.num:
            p = []
            for sub_dataset in self.all_dataset:
                sub_p = sub_dataset.pick
                p += sub_p
            np.random.shuffle(p)
            pick += p
            m = len(pick)
        logger.info("shuffle done!")
        logger.info("dataset length {}".format(self.num))
        return pick[:self.num]

    def _find_dataset(self, index):
        for dataset in self.all_dataset:
            if dataset.start_idx + dataset.num > index:
                return dataset, index - dataset.start_idx

    def _get_bbox(self, image, shape):
        imh, imw = image.shape[:2]
        if len(shape) == 4:
            w, h = shape[2]-shape[0], shape[3]-shape[1]
        else:
            w, h = shape
        context_amount = 0.5
        exemplar_size = cfg.TRAIN.EXEMPLAR_SIZE
        wc_z = w + context_amount * (w+h)
        hc_z = h + context_amount * (w+h)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = exemplar_size / s_z
        w = w*scale_z
        h = h*scale_z
        cx, cy = imw//2, imh//2
        bbox = center2corner(Center(cx, cy, w, h))
        return bbox

    def X2Cube(self, img,B=[4, 4],skip = [4, 4],bandNumber=16):
    # Parameters
        M, N = img.shape
        col_extent = N - B[1] + 1
        row_extent = M - B[0] + 1
        # Get Starting block indices
        start_idx = np.arange(B[0])[:, None] * N + np.arange(B[1])
        # Generate Depth indeces
        didx = M * N * np.arange(1)
        start_idx = (didx[:, None] + start_idx.ravel()).reshape((-1, B[0], B[1]))
        # Get offsetted indices across the height and width of input array
        offset_idx = np.arange(row_extent)[:, None] * N + np.arange(col_extent)
        # Get all actual indices & index into input array for final output
        out = np.take(img, start_idx.ravel()[:, None] + offset_idx[::skip[0], ::skip[1]].ravel())
        out = np.transpose(out)
        DataCube = out.reshape(M//4, N//4,bandNumber )
        return DataCube
    
    def crop_hwc(self, image, bbox, out_sz, padding=(0, 0, 0)):
        a = (out_sz-1) / (bbox[2]-bbox[0])
        b = (out_sz-1) / (bbox[3]-bbox[1])
        c = -a * bbox[0]
        d = -b * bbox[1]
        mapping = np.array([[a, 0, c],
                            [0, b, d]]).astype(np.float)
        crop = [cv2.warpAffine(image[:, :, i], 
                               mapping, 
                               (out_sz, out_sz), 
                               borderMode=cv2.BORDER_CONSTANT, 
                               borderValue=(float(padding[i]),float(padding[i]),float(padding[i]))) for i in range(image.shape[2])]
        return np.stack(crop, axis=0)


    def pos_s_2_bbox(self, pos, s):
        return [pos[0]-s/2, pos[1]-s/2, pos[0]+s/2, pos[1]+s/2]


    def crop_like_SiamFC(self, image, bbox, context_amount=0.5, exemplar_size=127, instanc_size=255, padding=(0, 0, 0)):
        bbox = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]
        target_pos = [(bbox[2]+bbox[0])/2., (bbox[3]+bbox[1])/2.]
        target_size = [bbox[2]-bbox[0], bbox[3]-bbox[1]]   # width, height
        wc_z = target_size[1] + context_amount * sum(target_size)
        hc_z = target_size[0] + context_amount * sum(target_size)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = exemplar_size / s_z
        d_search = (instanc_size - exemplar_size) / 2
        pad = d_search / scale_z
        s_x = s_z + 2 * pad

        z = self.crop_hwc(image, self.pos_s_2_bbox(target_pos, s_z), exemplar_size, padding=np.mean(image, axis=(0, 1)))
        x = self.crop_hwc(image, self.pos_s_2_bbox(target_pos, s_x), instanc_size, padding=np.mean(image, axis=(0, 1)))
        return z, x


    def __len__(self):
        return self.num

    def __getitem__(self, index):
        index = self.pick[index]
        dataset, index = self._find_dataset(index)

        gray = cfg.DATASET.GRAY and cfg.DATASET.GRAY > np.random.random()
        neg = cfg.DATASET.NEG and cfg.DATASET.NEG > np.random.random()

        # get one dataset
        if neg:
            template = dataset.get_random_target(index)
            search = np.random.choice(self.all_dataset).get_random_target()
        else:
            template, search = dataset.get_positive_pair(index)

        # get image
        search_HSI = template_HSI = None
        if dataset.name == 'Whispers':
            seq_name = template[0].split('\\')[-2]
            template_index = int(template[0].split('\\')[-1].split('.')[0]) + 1
            search_index = int(search[0].split('\\')[-1].split('.')[0]) + 1
            HSI_anno = np.loadtxt(join(cfg.DATASET.Whispers.BASEROOT, seq_name, 'HSI','groundtruth_rect.txt'))
            template_box = HSI_anno[template_index - 1]
            search_box = HSI_anno[search_index - 1]
            
            template_HSI = self.X2Cube(cv2.imread(join(cfg.DATASET.Whispers.BASEROOT, seq_name, 'HSI', '{:04d}.png'.format(template_index)), -1).astype(np.float32))
            template_HSI = self.crop_like_SiamFC(template_HSI, template_box)[0]
            search_HSI = self.X2Cube(cv2.imread(join(cfg.DATASET.Whispers.BASEROOT, seq_name, 'HSI', '{:04d}.png'.format(search_index)), -1).astype(np.float32))
            search_HSI = self.crop_like_SiamFC(search_HSI, search_box)[1]
        template_image = cv2.imread(template[0])
        search_image = cv2.imread(search[0])
        if template_image is None:
            print('error image:',template[0])

        # get bounding box
        template_box = self._get_bbox(template_image, template[1])
        search_box = self._get_bbox(search_image, search[1])


        # augmentation
        template, _ = self.template_aug(template_image,
                                        template_box,
                                        cfg.TRAIN.EXEMPLAR_SIZE,
                                        gray=gray)

        search, bbox = self.search_aug(search_image,
                                       search_box,
                                       cfg.TRAIN.SEARCH_SIZE,
                                       gray=gray)


        cls = np.zeros((cfg.TRAIN.OUTPUT_SIZE, cfg.TRAIN.OUTPUT_SIZE), dtype=np.int64)
        template = template.transpose((2, 0, 1)).astype(np.float32)
        search = search.transpose((2, 0, 1)).astype(np.float32)

        return {
                'template': template,
                'search': search,
                'label_cls': cls,
                'bbox': np.array([bbox.x1,bbox.y1,bbox.x2,bbox.y2]),
                'template_HSI': template_HSI,
                'search_HSI': search_HSI
                }

