from __future__ import division, print_function
import numpy as np

body25_to_joint = np.array([11, 10, 9, 12, 13, 14, 4, 3, 2, 5, 6, 7, -1,
                            -1, -1, -1, -1, -1, -1, 0, 16, 15, 18, 17], dtype=np.int32)
cam_f = 5000
img_res = 512
cam_c = img_res/2
cam_R = np.eye(3, dtype=np.float32) * np.array([[1, -1, -1]], dtype=np.float32)
cam_tz = 10.0
cam_t = np.array([[0, 0, cam_tz]], dtype=np.float32)

vol_res = 128
semantic_encoding_sigma = 0.05
smooth_kernel_size = 7

cmr_num_layers = 5
cmr_num_channels = 256

training_list_easy_fname = 'training_models_easy.txt'
training_list_hard_fname = 'training_models_hard.txt'

dataset_image_subfolder = 'image_data'
dataset_mesh_subfolder = 'mesh_data'

