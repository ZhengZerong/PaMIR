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
import scipy.io as sio
import datetime
import glob
import logging
import math

from util.base_trainer import BaseTrainer
from dataloader.dataloader_tex import TrainingImgDataset
from network.arch import PamirNet, TexPamirNetAttention
from neural_voxelization_layer.smpl_model import TetraSMPL
from neural_voxelization_layer.voxelize import Voxelization
from util.img_normalization import ImgNormalizerForResnet
from graph_cmr.models import GraphCNN, SMPLParamRegressor, SMPL
from graph_cmr.utils import Mesh
from graph_cmr.models.geometric_layers import rodrigues, orthographic_projection
import util.obj_io as obj_io
import util.util as util
import constant as const


class Trainer(BaseTrainer):
    def __init__(self, options):
        super(Trainer, self).__init__(options)

    def init_fn(self):
        super(BaseTrainer, self).__init__()
        # dataset
        self.train_ds = TrainingImgDataset(
            self.options.dataset_dir, img_h=const.img_res, img_w=const.img_res,
            training=True, testing_res=256,
            view_num_per_item=self.options.view_num_per_item,
            point_num=self.options.point_num,
            load_pts2smpl_idx_wgt=True,
            smpl_data_folder='./data')

        # neural voxelization components
        self.smpl = SMPL('./data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl').to(self.device)
        self.tet_smpl = TetraSMPL('./data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl',
                                  './data/tetra_smpl.npz').to(self.device)
        smpl_vertex_code, smpl_face_code, smpl_faces, smpl_tetras = util.read_smpl_constants('./data')
        self.smpl_faces = smpl_faces
        self.voxelization = Voxelization(smpl_vertex_code, smpl_face_code, smpl_faces, smpl_tetras,
                                         volume_res=const.vol_res,
                                         sigma=const.semantic_encoding_sigma,
                                         smooth_kernel_size=const.smooth_kernel_size,
                                         batch_size=self.options.batch_size).to(self.device)

        # pamir_net
        self.pamir_net = PamirNet().to(self.device)
        self.pamir_tex_net = TexPamirNetAttention().to(self.device)

        # optimizers
        self.optm_pamir_tex_net = torch.optim.Adam(
            params=list(self.pamir_tex_net.parameters()), lr=float(self.options.lr)
        )

        # loses
        self.criterion_tex = nn.L1Loss().to(self.device)

        # Pack models and optimizers in a dict - necessary for checkpointing
        self.models_dict = {'pamir_tex_net': self.pamir_tex_net}
        self.optimizers_dict = {'optimizer_pamir_net': self.optm_pamir_tex_net}

        # Optionally start training from a pretrained checkpoint
        # Note that this is different from resuming training
        # For the latter use --resume
        assert self.options.pretrained_pamir_net_checkpoint is not None, 'You must provide a pretrained PaMIR geometry model!'
        self.load_pretrained_pamir_net(self.options.pretrained_pamir_net_checkpoint)

        # read energy weights
        self.loss_weights = {
            'tex': 1.0,
            'att': 0.005,
        }

        logging.info('#trainable_params = %d' %
                     sum(p.numel() for p in self.pamir_tex_net.parameters() if p.requires_grad))

        # meta results
        now = datetime.datetime.now()
        self.log_file_path = os.path.join(
            self.options.log_dir, 'log_%s.npz' % now.strftime('%Y_%m_%d_%H_%M_%S'))

    def train_step(self, input_batch):
        self.pamir_tex_net.train()
        self.pamir_net.eval()

        # some constants
        cam_f, cam_tz, cam_c = const.cam_f, const.cam_tz, const.cam_c
        cam_r = torch.tensor([1, -1, -1], dtype=torch.float32).to(self.device)
        cam_t = torch.tensor([0, 0, cam_tz], dtype=torch.float32).to(self.device)

        # training data
        img = input_batch['img']
        pts = input_batch['pts']    # [:, :-self.options.point_num]
        pts_proj = input_batch['pts_proj']  # [:, :-self.options.point_num]
        gt_clr = input_batch['pts_clr']   # [:, :-self.options.point_num]
        gt_betas = input_batch['betas']
        gt_pose = input_batch['pose']
        gt_scale = input_batch['scale']
        gt_trans = input_batch['trans']

        batch_size, pts_num = pts.size()[:2]
        losses = dict()

        with torch.no_grad():
            gt_vert_cam = gt_scale * self.tet_smpl(gt_pose, gt_betas) + gt_trans
            vol = self.voxelization(gt_vert_cam)    # we simply use ground-truth SMPL for when training texture module
            img_feat_geo = self.pamir_net.get_img_feature(img, no_grad=True)

        output_clr_, output_clr, output_att, smpl_feat = self.pamir_tex_net.forward(
            img, vol, pts, pts_proj, img_feat_geo)
        losses['tex'] = self.tex_loss(output_clr, gt_clr) + self.tex_loss(output_clr_, gt_clr)
        losses['att'] = self.attention_loss(output_att)

        # calculates total loss
        total_loss = 0.
        for ln in losses.keys():
            w = self.loss_weights[ln] if ln in self.loss_weights else 1.0
            total_loss += w * losses[ln]
        losses.update({'total_loss': total_loss})

        # Do backprop
        self.optm_pamir_tex_net.zero_grad()
        total_loss.backward()
        self.optm_pamir_tex_net.step()

        # save
        self.write_logs(losses)

        # update learning rate
        if self.step_count % 10000 == 0:
            learning_rate = self.options.lr * (0.9 ** (self.step_count//10000))
            logging.info('Epoch %d, LR = %f' % (self.step_count, learning_rate))
            for param_group in self.optm_pamir_tex_net.param_groups:
                param_group['lr'] = learning_rate
        return losses

    def calculate_gt_rotmat(self, gt_pose):
        gt_rotmat = rodrigues(gt_pose.view((-1, 3)))
        gt_rotmat = gt_rotmat.view((self.options.batch_size, -1, 3, 3))
        gt_rotmat[:, 0, 1:3, :] = gt_rotmat[:, 0, 1:3, :] * -1.0  # global rotation
        return gt_rotmat

    def tex_loss(self, pred_clr, gt_clr, att=None):
        """Computes per-sample loss of the occupancy value"""
        if att is None:
            att = torch.ones_like(gt_clr)
        if len(att.size()) != len(gt_clr.size()):
            att = att.unsqueeze(0)
        loss = self.criterion_tex(pred_clr * att, gt_clr * att)
        return loss

    def attention_loss(self, pred_att):
        return torch.mean(-torch.log(pred_att + 1e-4))
        # return torch.mean((pred_att - 0.9) ** 2)

    def train_summaries(self, input_batch, losses=None):
        assert losses is not None
        for ln in losses.keys():
            self.summary_writer.add_scalar(ln, losses[ln].item(), self.step_count)

    def write_logs(self, losses):
        data = dict()
        if os.path.exists(self.log_file_path):
            data = dict(np.load(self.log_file_path))
            for k in losses.keys():
                data[k] = np.append(data[k], losses[k].item())
        else:
            for k in losses.keys():
                data[k] = np.array([losses[k].item()])
        np.savez(self.log_file_path, **data)

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
