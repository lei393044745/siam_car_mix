# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch.nn.functional as F
import cv2
import torch
from pysot.core.config import cfg
from pysot.tracker.base_tracker import SiameseTracker
from pysot.utils.misc import bbox_clip


class SiamCARTracker(SiameseTracker):
    def __init__(self, model, cfg):
        super(SiamCARTracker, self).__init__()
        hanning = np.hanning(cfg.SCORE_SIZE)
        self.window = np.outer(hanning, hanning)
        self.model = model
        self.model.eval()

    def _convert_cls(self, cls):
        cls = F.softmax(cls[:,:,:,:], dim=1).data[:,1,:,:].cpu().numpy()
        return cls

    def X2Cube(self, img,B=[4, 4],skip = [4, 4],bandNumber=16):
        img = cv2.imread(img, -1)
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
    

    def init(self, img, bbox, hsi, hsi_box):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """
        img = cv2.imread(img)
        hsi_img = self.X2Cube(hsi).astype(np.float32)
        # hsi_img = torch.from_numpy(self.crop_like_SiamFC(hsi_img, hsi_box)[0]).cuda()
        self.center_pos = np.array([bbox[0]+(bbox[2]-1)/2,
                                    bbox[1]+(bbox[3]-1)/2])
        self.size = np.array([bbox[2], bbox[3]])

        # calculate z crop size
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average = np.mean(img, axis=(0, 1))
        self.hsi_channel_average = np.mean(hsi_img, axis=(0, 1))
        
        # get crop
        z_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, self.channel_average)
        hsi_img = self.get_subwindow(hsi_img, self.center_pos,
                                     cfg.TRACK.EXEMPLAR_SIZE,
                                     s_z, self.hsi_channel_average)
        self.model.template(z_crop)
        self.model.extract_his(hsi_img)

    def change(self,r):
        return np.maximum(r, 1. / r)

    def sz(self, w, h):
        pad = (w + h) * 0.5
        return np.sqrt((w + pad) * (h + pad))

    def cal_penalty(self, lrtbs, penalty_lk):
        bboxes_w = lrtbs[0, :, :] + lrtbs[2, :, :]
        bboxes_h = lrtbs[1, :, :] + lrtbs[3, :, :]
        s_c = self.change(self.sz(bboxes_w, bboxes_h) / self.sz(self.size[0]*self.scale_z, self.size[1]*self.scale_z))
        r_c = self.change((self.size[0] / self.size[1]) / (bboxes_w / bboxes_h))
        penalty = np.exp(-(r_c * s_c - 1) * penalty_lk)
        return penalty

    def accurate_location(self, max_r_up, max_c_up):
        dist = int((cfg.TRACK.INSTANCE_SIZE - (cfg.TRACK.SCORE_SIZE - 1) * 8) / 2)
        max_r_up += dist
        max_c_up += dist
        p_cool_s = np.array([max_r_up, max_c_up])
        disp = p_cool_s - (np.array([cfg.TRACK.INSTANCE_SIZE, cfg.TRACK.INSTANCE_SIZE]) - 1.) / 2.
        return disp

    def coarse_location(self, hp_score_up, p_score_up, scale_score, lrtbs):
        upsize = (cfg.TRACK.SCORE_SIZE - 1) * cfg.TRACK.STRIDE + 1
        max_r_up_hp, max_c_up_hp = np.unravel_index(hp_score_up.argmax(), hp_score_up.shape)
        max_r = int(round(max_r_up_hp / scale_score))
        max_c = int(round(max_c_up_hp / scale_score))
        max_r = bbox_clip(max_r, 0, cfg.TRACK.SCORE_SIZE)
        max_c = bbox_clip(max_c, 0, cfg.TRACK.SCORE_SIZE)
        bbox_region = lrtbs[max_r, max_c, :]
        min_bbox = int(cfg.TRACK.REGION_S * cfg.TRACK.EXEMPLAR_SIZE)
        max_bbox = int(cfg.TRACK.REGION_L * cfg.TRACK.EXEMPLAR_SIZE)
        l_region = int(min(max_c_up_hp, bbox_clip(bbox_region[0], min_bbox, max_bbox)) / 2.0)
        t_region = int(min(max_r_up_hp, bbox_clip(bbox_region[1], min_bbox, max_bbox)) / 2.0)

        r_region = int(min(upsize - max_c_up_hp, bbox_clip(bbox_region[2], min_bbox, max_bbox)) / 2.0)
        b_region = int(min(upsize - max_r_up_hp, bbox_clip(bbox_region[3], min_bbox, max_bbox)) / 2.0)
        mask = np.zeros_like(p_score_up)
        mask[max_r_up_hp - t_region:max_r_up_hp + b_region + 1, max_c_up_hp - l_region:max_c_up_hp + r_region + 1] = 1
        p_score_up = p_score_up * mask
        return p_score_up

    def getCenter(self,hp_score_up, p_score_up, scale_score,lrtbs):
        # corse location
        score_up = self.coarse_location(hp_score_up, p_score_up, scale_score, lrtbs)
        # accurate location
        max_r_up, max_c_up = np.unravel_index(score_up.argmax(), score_up.shape)
        disp = self.accurate_location(max_r_up,max_c_up)
        disp_ori = disp / self.scale_z
        new_cx = disp_ori[1] + self.center_pos[0]
        new_cy = disp_ori[0] + self.center_pos[1]
        return max_r_up, max_c_up, new_cx, new_cy

    def track(self, img, hp, hsi_img):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        hsi_img = self.X2Cube(hsi_img)
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        self.scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        x_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average)
        hsi_x_crop = self.get_subwindow(hsi_img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.hsi_channel_average)

        outputs = self.model.track(x_crop, hsi_x_crop)
        cls = self._convert_cls(outputs['cls']).squeeze()
        cen = outputs['cen'].data.cpu().numpy()
        cen = (cen - cen.min()) / cen.ptp()
        cen = cen.squeeze()
        lrtbs = outputs['loc'].data.cpu().numpy().squeeze()

        upsize = (cfg.TRACK.SCORE_SIZE-1) * cfg.TRACK.STRIDE + 1
        penalty = self.cal_penalty(lrtbs, hp['penalty_k'])
        p_score = penalty * cls * cen
        if cfg.TRACK.hanming:
            hp_score = p_score*(1 - hp['window_lr']) + self.window * hp['window_lr']
        else:
            hp_score = p_score

        hp_score_up = cv2.resize(hp_score, (upsize, upsize), interpolation=cv2.INTER_CUBIC)
        p_score_up = cv2.resize(p_score, (upsize, upsize), interpolation=cv2.INTER_CUBIC)
        cls_up = cv2.resize(cls, (upsize, upsize), interpolation=cv2.INTER_CUBIC)
        lrtbs = np.transpose(lrtbs,(1,2,0))
        lrtbs_up = cv2.resize(lrtbs, (upsize, upsize), interpolation=cv2.INTER_CUBIC)

        scale_score = upsize / cfg.TRACK.SCORE_SIZE
        # get center
        max_r_up, max_c_up, new_cx, new_cy = self.getCenter(hp_score_up, p_score_up, scale_score, lrtbs)
        # get w h
        ave_w = (lrtbs_up[max_r_up,max_c_up,0] + lrtbs_up[max_r_up,max_c_up,2]) / self.scale_z
        ave_h = (lrtbs_up[max_r_up,max_c_up,1] + lrtbs_up[max_r_up,max_c_up,3]) / self.scale_z

        s_c = self.change(self.sz(ave_w, ave_h) / self.sz(self.size[0]*self.scale_z, self.size[1]*self.scale_z))
        r_c = self.change((self.size[0] / self.size[1]) / (ave_w / ave_h))
        penalty = np.exp(-(r_c * s_c - 1) * hp['penalty_k'])
        lr = penalty * cls_up[max_r_up, max_c_up] * hp['lr']
        new_width = lr*ave_w + (1-lr)*self.size[0]
        new_height = lr*ave_h + (1-lr)*self.size[1]

        # clip boundary
        cx = bbox_clip(new_cx,0,img.shape[1])
        cy = bbox_clip(new_cy,0,img.shape[0])
        width = bbox_clip(new_width,0,img.shape[1])
        height = bbox_clip(new_height,0,img.shape[0])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])
        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]

        return {
                'bbox': bbox,
               }
