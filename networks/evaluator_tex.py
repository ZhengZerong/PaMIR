"""
This file includes the full training procedure.
"""
from __future__ import print_function
from __future__ import division

import os
import numpy as np
import cv2 as cv
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import measure
from tqdm import tqdm
import scipy.io as sio
import datetime
import glob
import logging
import math

from network.arch import PamirNet, TexPamirNetAttention
from neural_voxelization_layer.voxelize import Voxelization
from neural_voxelization_layer.smpl_model import TetraSMPL
from util.img_normalization import ImgNormalizerForResnet
from graph_cmr.models import GraphCNN, SMPLParamRegressor2, SMPL
from graph_cmr.utils import Mesh
import util.util as util
import util.obj_io as obj_io
import constant as const


class EvaluatorTex(object):
    def __init__(self, device, pretrained_checkpoint_pamir, pretrained_checkpoint_pamir_tex):
        super(EvaluatorTex, self).__init__()
        util.configure_logging(True, False, None)

        self.device = device

        # neural voxelization components
        self.smpl = SMPL('./data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl').to(self.device)
        self.tet_smpl = TetraSMPL('./data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl',
                                  './data/tetra_smpl.npz').to(self.device)
        smpl_vertex_code, smpl_face_code, smpl_faces, smpl_tetras = \
            util.read_smpl_constants('./data')
        self.smpl_faces = smpl_faces
        self.voxelization = Voxelization(smpl_vertex_code, smpl_face_code, smpl_faces, smpl_tetras,
                                         volume_res=const.vol_res,
                                         sigma=const.semantic_encoding_sigma,
                                         smooth_kernel_size=const.smooth_kernel_size,
                                         batch_size=1).to(self.device)
        # pamir_net
        self.pamir_net = PamirNet().to(self.device)
        self.pamir_tex_net = TexPamirNetAttention().to(self.device)

        self.models_dict = {'pamir_tex_net': self.pamir_tex_net}
        self.load_pretrained_pamir_net(pretrained_checkpoint_pamir)
        self.load_pretrained(checkpoint_file=pretrained_checkpoint_pamir_tex)
        self.pamir_net.eval()
        self.pamir_tex_net.eval()

    def load_pretrained(self, checkpoint_file=None):
        """Load a pretrained checkpoint.
        This is different from resuming training using --resume.
        """
        if checkpoint_file is not None:
            checkpoint = torch.load(checkpoint_file)
            for model in self.models_dict:
                if model in checkpoint:
                    self.models_dict[model].load_state_dict(checkpoint[model])
                    logging.info('Loading pamir_tex_net from ' + checkpoint_file)

    def test_tex_pifu(self, img, mesh_v, betas, pose, scale, trans):
        self.pamir_net.eval()
        self.pamir_tex_net.eval()
        gt_vert_cam = scale * self.tet_smpl(pose, betas) + trans
        vol = self.voxelization(gt_vert_cam)

        group_size = 512 * 128
        cam_f, cam_tz, cam_c = const.cam_f, const.cam_tz, const.cam_c
        cam_r = torch.tensor([1, -1, -1], dtype=torch.float32).to(self.device)
        cam_t = torch.tensor([0, 0, cam_tz], dtype=torch.float32).to(self.device)

        pts = mesh_v
        pts_proj = self.forward_project_points(
            pts, cam_r, cam_t, cam_f, img.size(2))
        clr = self.forward_infer_color_value_group(
            img, vol, pts, pts_proj, group_size)
        return clr

    def test_att_pifu(self, img, mesh_v, betas, pose, scale, trans):
        self.pamir_net.eval()
        self.pamir_tex_net.eval()
        gt_vert_cam = scale * self.tet_smpl(pose, betas) + trans
        vol = self.voxelization(gt_vert_cam)
        group_size = 512 * 128
        cam_f, cam_tz, cam_c = const.cam_f, const.cam_tz, const.cam_c
        cam_r = torch.tensor([1, -1, -1], dtype=torch.float32).to(self.device)
        cam_t = torch.tensor([0, 0, cam_tz], dtype=torch.float32).to(self.device)

        pts = mesh_v
        pts_proj = self.forward_project_points(
            pts, cam_r, cam_t, cam_f, img.size(2))
        att = self.forward_infer_attention_value_group(
            img, vol, pts, pts_proj, group_size)
        att = np.squeeze(att).reshape((-1, 1))
        att = np.concatenate([att, att, att], axis=1)
        return att

    def forward_project_points(self, pts, cam_r, cam_t, cam_f, img_res):
        pts_proj = pts * cam_r.view((1, 1, -1)) + cam_t.view((1, 1, -1))
        pts_proj = pts_proj * (cam_f / (img_res / 2)) / pts_proj[:, :, 2:3]
        pts_proj = pts_proj[:, :, :2]
        return pts_proj

    def forward_infer_color_value_group(self, img, vol, pts, pts_proj, group_size):
        pts_group_num = (pts.size()[1] + group_size - 1) // group_size
        pts_clr = []
        for gi in tqdm(range(pts_group_num), desc='Texture query'):
            # print('Testing point group: %d/%d' % (gi + 1, pts_group_num))
            pts_group = pts[:, (gi * group_size):((gi + 1) * group_size), :]
            pts_proj_group = pts_proj[:, (gi * group_size):((gi + 1) * group_size), :]
            outputs = self.forward_infer_color_value(
                img, vol, pts_group, pts_proj_group)
            pts_clr.append(np.squeeze(outputs[0].detach().cpu().numpy()))
        pts_clr = np.concatenate(pts_clr)
        pts_clr = np.array(pts_clr)
        return pts_clr

    def forward_infer_color_value(self, img, vol, pts, pts_proj):
        img_feat_geo = self.pamir_net.get_img_feature(img, no_grad=True)
        _, clr, _, _ = self.pamir_tex_net.forward(img, vol, pts, pts_proj, img_feat_geo)
        return clr

    def forward_infer_attention_value_group(self, img, vol, pts, pts_proj, group_size):
        pts_group_num = (pts.size()[1] + group_size - 1) // group_size
        pts_att = []
        for gi in tqdm(range(pts_group_num), desc='Texture query'):
            # print('Testing point group: %d/%d' % (gi + 1, pts_group_num))
            pts_group = pts[:, (gi * group_size):((gi + 1) * group_size), :]
            pts_proj_group = pts_proj[:, (gi * group_size):((gi + 1) * group_size), :]
            outputs = self.forward_infer_attention_value(
                img, vol, pts_group, pts_proj_group)
            pts_att.append(np.squeeze(outputs.detach().cpu().numpy()))
        pts_att = np.concatenate(pts_att)
        pts_att = np.array(pts_att)
        return pts_att

    def forward_infer_attention_value(self, img, vol, pts, pts_proj):
        img_feat_geo = self.pamir_net.get_img_feature(img, no_grad=True)
        _, _, att, _ = self.pamir_tex_net.forward(img, vol, pts, pts_proj, img_feat_geo)
        return att

    def load_pretrained_pamir_net(self, model_path):
        if os.path.isdir(model_path):
            tmp1 = glob.glob(os.path.join(model_path, 'pamir_net*.pt'))
            assert len(tmp1) == 1
            logging.info('Loading pamir_net from ' + tmp1[0])
            data = torch.load(tmp1[0])
        else:
            logging.info('Loading pamir_net from ' + model_path)
            data = torch.load(model_path)
        if 'pamir_net' in data:
            self.pamir_net.load_state_dict(data['pamir_net'])
        else:
            raise IOError('Failed to load pamir_net model from the specified checkpoint!!')
