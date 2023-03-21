# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

from pysot.core.config import cfg
from pysot.models.loss_car import make_siamcar_loss_evaluator
from pysot.models.backbone import get_backbone
from pysot.models.head.car_head import CARHead
from pysot.models.neck import get_neck
from ..utils.location_grid import compute_locations
from pysot.utils.xcorr import xcorr_depthwise


class HSI_backbone(nn.Module):
    def __init__(self):
        super(HSI_backbone, self).__init__()
        self.Block1 = nn.Sequential(
                nn.Conv3d(in_channels=1, out_channels=256,kernel_size=(3,3,3), stride=(2,2,2), bias=True),
                nn.BatchNorm3d(256),
                nn.ReLU(inplace=True),
                nn.Conv3d(in_channels=256, out_channels=256,kernel_size=(1,3,3), stride=(1,2,2), bias=True),
                nn.BatchNorm3d(256),
                nn.ReLU(inplace=True),
                nn.Conv3d(in_channels=256, out_channels=256,kernel_size=(3,3,3), stride=(2,2,2), bias=True),
                nn.BatchNorm3d(256),
                nn.ReLU(inplace=True),
            
        )
        self.conv2_1 = nn.Sequential(
            nn.Conv3d(in_channels=256, out_channels=256, kernel_size=(1,1,1), stride=(1,1,1),bias=True),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True)
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv3d(in_channels=256, out_channels=256, kernel_size=(3,1,1), stride=(1,1,1),bias=True),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True)
        )
        
        self.conv3_1 = nn.Sequential(
            nn.Conv3d(in_channels=256, out_channels=256, kernel_size=(1,1,1), stride=(1,1,1),bias=True),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True)
        )
        self.conv3_2 = nn.Sequential(
            nn.Conv3d(in_channels=256, out_channels=256, kernel_size=(3,1,1), stride=(1,1,1),bias=True),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True)
        )



    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.Block1(x)
        x1 = self.conv2_1(x) + self.conv2_2(x)
        x2 = self.conv3_1(x1) + self.conv3_2(x1)
        if x1.size(3) < 20:
            x1 = x1[:, :, :,4:11, 4:11]
            x2 = x2[:, :, :,4:11, 4:11]
        return [x1,x2]


class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbon
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)
        # build adjust layer
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)

        # build car head
        self.car_head = CARHead(cfg, 256)

        # build response map
        self.xcorr_depthwise = xcorr_depthwise

        # build loss
        self.loss_evaluator = make_siamcar_loss_evaluator(cfg)

        self.down = nn.ConvTranspose2d(256 * 3, 256, 1, 1)

        self.backbone_HSI = HSI_backbone()

        self.h_down = nn.ConvTranspose2d(256 * 6, 256, 1, 1)

    def template(self, z):
        zf = self.backbone(z)
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
        self.zf = zf

    def track(self, x):
        xf = self.backbone(x)
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)

        features = self.xcorr_depthwise(xf[0],self.zf[0])
        for i in range(len(xf)-1):
            features_new = self.xcorr_depthwise(xf[i+1],self.zf[i+1])
            features = torch.cat([features,features_new],1)
        features = self.down(features)

        cls, loc, cen = self.car_head(features)
        return {
                'cls': cls,
                'loc': loc,
                'cen': cen
               }

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    def forward(self, data):
        """ only used in training
        """
        template = data['template'].cuda()
        search = data['search'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['bbox'].cuda()
        template_HSI = data['template_HSI'].cuda()
        search_HSI = data['search_HSI'].cuda()
        HSI_zf = self.backbone_HSI(template_HSI)
        HSI_xf = self.backbone_HSI(search_HSI)
        HSI_features = []
        for i in range(len(HSI_xf)):
            for j in range(HSI_xf[i].size(2)):
                HSI_features.append(self.xcorr_depthwise(HSI_xf[i][:,:,j,:], HSI_zf[i][:,:,j,:]))
        # get feature
        h_features = torch.cat(HSI_features, dim=1)
        h_features = self.h_down(h_features)
        zf = self.backbone(template)
        xf = self.backbone(search)
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
            xf = self.neck(xf)


        features = self.xcorr_depthwise(xf[0],zf[0])
        for i in range(len(xf)-1):
            features_new = self.xcorr_depthwise(xf[i+1],zf[i+1])
            features = torch.cat([features,features_new],1)
        features = self.down(features)

        cls, loc, cen = self.car_head(features)
        locations = compute_locations(cls, cfg.TRACK.STRIDE)
        cls = self.log_softmax(cls)
        cls_loss, loc_loss, cen_loss = self.loss_evaluator(
            locations,
            cls,
            loc,
            cen, label_cls, label_loc
        )

        # get loss
        outputs = {}
        outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
            cfg.TRAIN.LOC_WEIGHT * loc_loss + cfg.TRAIN.CEN_WEIGHT * cen_loss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss
        outputs['cen_loss'] = cen_loss
        return outputs
