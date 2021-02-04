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

from network.arch import PamirNet
from neural_voxelization_layer.smpl_model import TetraSMPL
from neural_voxelization_layer.voxelize import Voxelization
from util.img_normalization import ImgNormalizerForResnet
from graph_cmr.models import GraphCNN, SMPLParamRegressor, SMPL
from graph_cmr.utils import Mesh
from graph_cmr.models.geometric_layers import rodrigues, orthographic_projection
import util.util as util
import util.obj_io as obj_io
import constant as const


class Evaluator(object):
    def __init__(self, device, pretrained_checkpoint, gcmr_checkpoint):
        super(Evaluator, self).__init__()
        util.configure_logging(True, False, None)

        self.device = device

        # GraphCMR components
        self.img_norm = ImgNormalizerForResnet().to(self.device)
        self.graph_mesh = Mesh()
        self.graph_cnn = GraphCNN(self.graph_mesh.adjmat, self.graph_mesh.ref_vertices.t(),
                                  const.cmr_num_layers, const.cmr_num_channels).to(self.device)
        self.smpl_param_regressor = SMPLParamRegressor().to(self.device)

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

        # PIFU
        self.pamir_net = PamirNet().to(self.device)
        self.models_dict = {'pamir_net': self.pamir_net}
        self.load_pretrained(checkpoint_file=pretrained_checkpoint)
        self.load_pretrained_gcmr(gcmr_checkpoint)
        self.graph_cnn.eval()
        self.smpl_param_regressor.eval()
        self.pamir_net.eval()

    def load_pretrained(self, checkpoint_file=None):
        """Load a pretrained checkpoint.
        This is different from resuming training using --resume.
        """
        if checkpoint_file is not None:
            checkpoint = torch.load(checkpoint_file)
            for model in self.models_dict:
                if model in checkpoint:
                    logging.info('Loading %s from %s' % (model, checkpoint_file))
                    self.models_dict[model].load_state_dict(checkpoint[model])

    def load_pretrained_gcmr(self, model_path):
        if os.path.isdir(model_path):
            tmp = glob.glob(os.path.join(model_path, 'gcmr*.pt'))
            assert len(tmp) == 1
            logging.info('Loading GraphCMR from ' + tmp[0])
            data = torch.load(tmp[0])
        else:
            logging.info('Loading GraphCMR from ' + model_path)
            data = torch.load(model_path)
        self.graph_cnn.load_state_dict(data['graph_cnn'])
        self.smpl_param_regressor.load_state_dict(data['smpl_param_regressor'])

    def test_gcmr(self, img):
        self.graph_cnn.eval()
        self.smpl_param_regressor.eval()

        # gcmr body prediction
        pred_cam, pred_rotmat, pred_betas, pred_vert_sub, \
        pred_vert, pred_vert_tetsmpl = self.forward_gcmr(img)

        # camera coordinate conversion
        cam_f, cam_tz, cam_c = const.cam_f, const.cam_tz, const.cam_c
        scale_, trans_ = self.forward_coordinate_conversion(cam_f, cam_tz, cam_c, pred_cam)
        return pred_betas, pred_rotmat, scale_, trans_, pred_vert_tetsmpl[:, :6890]

    def test_pifu(self, img, vol_res, betas, pose, scale, trans):
        self.pamir_net.train()
        self.graph_cnn.eval()  # lock BN and dropout
        self.smpl_param_regressor.eval()  # lock BN and dropout
        gt_vert_cam = scale * self.tet_smpl(pose, betas) + trans
        vol = self.voxelization(gt_vert_cam)
        group_size = 512 * 80
        grid_ov = self.forward_infer_occupancy_value_grid_octree(img, vol, vol_res, group_size)
        vertices, simplices, normals, _ = measure.marching_cubes_lewiner(grid_ov, 0.5)

        mesh = dict()
        mesh['v'] = vertices / vol_res - 0.5
        mesh['f'] = simplices[:, (1, 0, 2)]
        mesh['vn'] = normals
        return mesh

    def optm_smpl_param(self, img, keypoint, betas, pose, scale, trans, iter_num):
        assert iter_num > 0
        self.pamir_net.eval()
        cam_f, cam_tz, cam_c = const.cam_f, const.cam_tz, const.cam_c
        cam_r = torch.tensor([1, -1, -1], dtype=torch.float32).to(self.device)
        cam_t = torch.tensor([0, 0, cam_tz], dtype=torch.float32).to(self.device)
        kp_conf = keypoint[:, :, -1:].clone()
        kp_detection = keypoint[:, :, :-1].clone()

        # convert rotmat to theta
        rotmat_host = pose.detach().cpu().numpy().squeeze()
        theta_host = []
        for r in rotmat_host:
            theta_host.append(cv.Rodrigues(r)[0])
        theta_host = np.asarray(theta_host).reshape((1, -1))
        theta = torch.from_numpy(theta_host).to(self.device)

        # construct parameters
        vert_cam = scale * self.tet_smpl(theta, betas) + trans
        vol = self.voxelization(vert_cam)
        theta_new = torch.nn.Parameter(theta)
        betas_new = torch.nn.Parameter(betas)
        theta_orig = theta_new.clone().detach()
        betas_orig = betas_new.clone().detach()
        optm = torch.optim.Adam(params=(theta_new, ), lr=2e-3)
        vert_tetsmpl = self.tet_smpl(theta_orig, betas_orig)
        vert_tetsmpl_cam = scale * vert_tetsmpl + trans
        keypoint = self.smpl.get_joints(vert_tetsmpl_cam[:, :6890])
        for i in tqdm(range(iter_num), desc='Body Fitting Optimization'):
            theta_new_ = torch.cat([theta_orig[:, :3], theta_new[:, 3:]], dim=1)
            vert_tetsmpl_new = self.tet_smpl(theta_new_, betas_new)
            vert_tetsmpl_new_cam = scale * vert_tetsmpl_new + trans
            keypoint_new = self.smpl.get_joints(vert_tetsmpl_new_cam[:, :6890])
            keypoint_new_proj = self.forward_point_sample_projection(
                keypoint_new, cam_r, cam_t, cam_f, cam_c)

            if i % 20 == 0:
                vol = self.voxelization(vert_tetsmpl_new_cam.detach())

            pred_vert_new_cam = self.graph_mesh.downsample(vert_tetsmpl_new_cam[:, :6890], n2=1)
            pred_vert_new_proj = self.forward_point_sample_projection(
                pred_vert_new_cam, cam_r, cam_t, cam_f, cam_c)
            smpl_sdf = self.forward_infer_occupancy_value(
                img, pred_vert_new_cam, pred_vert_new_proj, vol)

            loss_fitting = torch.mean(torch.abs(F.leaky_relu(0.5 - smpl_sdf, negative_slope=0.5)))
            loss_bias = torch.mean((theta_orig - theta_new) ** 2) + \
                        torch.mean((betas_orig - betas_new) ** 2) * 0.01
            loss_kp = torch.mean((kp_conf * keypoint_new_proj - kp_conf * kp_detection) ** 2)
            loss_bias2 = torch.mean((keypoint[:, :, 2] - keypoint_new[:, :, 2]) ** 2)

            loss = loss_fitting * 1.0 + loss_bias * 1.0 + loss_kp * 500.0 + loss_bias2 * 5

            optm.zero_grad()
            loss.backward()
            optm.step()
            # print('Iter No.%d: loss_fitting = %f, loss_bias = %f, loss_kp = %f' %
            #       (i, loss_fitting.item(), loss_bias.item(), loss_kp.item()))

        return theta_new, betas_new, vert_tetsmpl_new_cam[:, :6890]

    def generate_point_grids(self, vol_res, cam_R, cam_t, cam_f, img_res):
        x_coords = np.array(range(0, vol_res), dtype=np.float32)
        y_coords = np.array(range(0, vol_res), dtype=np.float32)
        z_coords = np.array(range(0, vol_res), dtype=np.float32)

        yv, xv, zv = np.meshgrid(x_coords, y_coords, z_coords)
        xv = np.reshape(xv, (-1, 1))
        yv = np.reshape(yv, (-1, 1))
        zv = np.reshape(zv, (-1, 1))
        xv = xv / vol_res - 0.5 + 0.5 / vol_res
        yv = yv / vol_res - 0.5 + 0.5 / vol_res
        zv = zv / vol_res - 0.5 + 0.5 / vol_res
        pts = np.concatenate([xv, yv, zv], axis=-1)
        pts = np.float32(pts)
        pts_proj = np.dot(pts, cam_R.transpose()) + cam_t
        pts_proj[:, 0] = pts_proj[:, 0] * cam_f / pts_proj[:, 2] / (img_res / 2)
        pts_proj[:, 1] = pts_proj[:, 1] * cam_f / pts_proj[:, 2] / (img_res / 2)
        pts_proj = pts_proj[:, :2]

        return pts, pts_proj

    def forward_gcmr(self, img):
        # GraphCMR forward
        batch_size = img.size()[0]
        img_ = self.img_norm(img)
        pred_vert_sub, pred_cam = self.graph_cnn(img_)
        pred_vert_sub = pred_vert_sub.transpose(1, 2)
        pred_vert = self.graph_mesh.upsample(pred_vert_sub)
        x = torch.cat(
            [pred_vert_sub, self.graph_mesh.ref_vertices[None, :, :].expand(batch_size, -1, -1)],
            dim=-1)
        pred_rotmat, pred_betas = self.smpl_param_regressor(x)
        pred_vert_tetsmpl = self.tet_smpl(pred_rotmat, pred_betas)
        return pred_cam, pred_rotmat, pred_betas, pred_vert_sub, \
               pred_vert, pred_vert_tetsmpl

    def forward_keypoint_projection(self, smpl_vert, cam):
        pred_keypoints = self.smpl.get_joints(smpl_vert)
        pred_keypoints_2d = orthographic_projection(pred_keypoints, cam)
        return pred_keypoints_2d

    def forward_coordinate_conversion(self, cam_f, cam_tz, cam_c, pred_cam):
        # calculates camera parameters
        with torch.no_grad():
            scale = pred_cam[:, 0:1] * cam_c * cam_tz / cam_f
            trans_x = pred_cam[:, 1:2] * cam_c * cam_tz * pred_cam[:, 0:1] / cam_f
            trans_y = -pred_cam[:, 2:3] * cam_c * cam_tz * pred_cam[:, 0:1] / cam_f
            trans_z = torch.zeros_like(trans_x)
            scale_ = torch.cat([scale, -scale, -scale], dim=-1).detach().view((-1, 1, 3))
            trans_ = torch.cat([trans_x, trans_y, trans_z], dim=-1).detach().view((-1, 1, 3))

        return scale_, trans_

    def forward_point_sample_projection(self, points, cam_r, cam_t, cam_f, cam_c):
        points_proj = points * cam_r.view((1, 1, -1)) + cam_t.view((1, 1, -1))
        points_proj = points_proj * (cam_f / cam_c) / points_proj[:, :, 2:3]
        points_proj = points_proj[:, :, :2]
        return points_proj

    def forward_infer_occupancy_value_grid_naive(self, img, vol, test_res, group_size):
        pts, pts_proj = self.generate_point_grids(
            test_res, const.cam_R, const.cam_t, const.cam_f, img.size(2))
        pts_ov = self.forward_infer_occupancy_value_group(img, vol, pts, pts_proj, group_size)
        pts_ov = pts_ov.reshape([test_res, test_res, test_res])
        return pts_ov

    def forward_infer_occupancy_value_grid_octree(self, img, vol, test_res, group_size,
                                             init_res=64, ignore_thres=0.05):
        pts, pts_proj = self.generate_point_grids(
            test_res, const.cam_R, const.cam_t, const.cam_f, img.size(2))
        pts = np.reshape(pts, (test_res, test_res, test_res, 3))
        pts_proj = np.reshape(pts_proj, (test_res, test_res, test_res, 2))

        pts_ov = np.zeros([test_res, test_res, test_res])
        dirty = np.ones_like(pts_ov, dtype=np.bool)
        grid_mask = np.zeros_like(pts_ov, dtype=np.bool)

        reso = test_res // init_res
        while reso > 0:
            grid_mask[0:test_res:reso, 0:test_res:reso, 0:test_res:reso] = True
            test_mask = np.logical_and(grid_mask, dirty)

            pts_ = pts[test_mask]
            pts_proj_ = pts_proj[test_mask]
            pts_ov[test_mask] = self.forward_infer_occupancy_value_group(
                img, vol, pts_, pts_proj_, group_size).squeeze()

            if reso <= 1:
                break
            for x in range(0, test_res - reso, reso):
                for y in range(0, test_res - reso, reso):
                    for z in range(0, test_res - reso, reso):
                        # if center marked, return
                        if not dirty[x + reso // 2, y + reso // 2, z + reso // 2]:
                            continue
                        v0 = pts_ov[x, y, z]
                        v1 = pts_ov[x, y, z + reso]
                        v2 = pts_ov[x, y + reso, z]
                        v3 = pts_ov[x, y + reso, z + reso]
                        v4 = pts_ov[x + reso, y, z]
                        v5 = pts_ov[x + reso, y, z + reso]
                        v6 = pts_ov[x + reso, y + reso, z]
                        v7 = pts_ov[x + reso, y + reso, z + reso]
                        v = np.array([v0, v1, v2, v3, v4, v5, v6, v7])
                        v_min = v.min()
                        v_max = v.max()
                        # this cell is all the same
                        if (v_max - v_min) < ignore_thres:
                            pts_ov[x:x + reso, y:y + reso, z:z + reso] = (v_max + v_min) / 2
                            dirty[x:x + reso, y:y + reso, z:z + reso] = False
            reso //= 2
        return pts_ov

    def forward_infer_occupancy_value_group(self, img, vol, pts, pts_proj, group_size):
        assert isinstance(pts, np.ndarray)
        assert len(pts.shape) == 2
        assert pts.shape[1] == 3
        pts_num = pts.shape[0]
        pts = torch.from_numpy(pts).unsqueeze(0).to(self.device)
        pts_proj = torch.from_numpy(pts_proj).unsqueeze(0).to(self.device)
        pts_group_num = (pts.size()[1] + group_size - 1) // group_size
        pts_ov = []
        for gi in tqdm(range(pts_group_num), desc='SDF query'):
            # print('Testing point group: %d/%d' % (gi + 1, pts_group_num))
            pts_group = pts[:, (gi * group_size):((gi + 1) * group_size), :]
            pts_proj_group = pts_proj[:, (gi * group_size):((gi + 1) * group_size), :]
            outputs = self.forward_infer_occupancy_value(
                img, pts_group, pts_proj_group, vol)
            pts_ov.append(np.squeeze(outputs.detach().cpu().numpy()))
        pts_ov = np.concatenate(pts_ov)
        pts_ov = np.array(pts_ov)
        return pts_ov

    def forward_infer_occupancy_value(self, img, pts, pts_proj, vol):
        return self.pamir_net(img, vol, pts, pts_proj)[-1]