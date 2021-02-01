import numpy as np
import os
import cv2 as cv
import glob
import math
import scipy.spatial
from tqdm import tqdm
import scipy.io as sio
import trimesh
import trimesh.sample
import trimesh.curvature
import multiprocessing

import objio

""" 
runtime configuration 
"""
mesh_data_dir = '../dataset_example/mesh_data'
output_data_dir = '../dataset_example/image_data'
view_num = 360
cam_f = 5000
cam_dist = 10


def get_data_list():
    """reads data list"""
    data_list = glob.glob(os.path.join(mesh_data_dir, './*/'))
    return sorted(data_list)


def read_data(item):
    """reads data """
    mesh_filename = glob.glob(os.path.join(item, '*.obj'))[0]  # assumes one .obj file
    mesh = trimesh.load(mesh_filename)
    return mesh


def process_one_data_item(data_item):
    print(data_item)
    _, item_name = os.path.split(data_item[:-1])
    output_fd = os.path.join(output_data_dir, item_name)
    os.makedirs(output_fd, exist_ok=True)
    os.makedirs(os.path.join(output_fd, 'sample'), exist_ok=True)

    smpl = objio.load_obj_data(os.path.join(mesh_data_dir, item_name, 'smpl/smpl_mesh.obj'))
    pts_data = sio.loadmat(os.path.join(output_fd, 'sample/samples.mat'))
    kd_tree = scipy.spatial.KDTree(smpl['v'])
    dist_surface_points_inside, idx_surface_points_inside = kd_tree.query(pts_data['surface_points_inside'], k=4)
    dist_surface_points_outside, idx_surface_points_outside = kd_tree.query(pts_data['surface_points_outside'], k=4)
    dist_uniform_points_inside, idx_uniform_points_inside = kd_tree.query(pts_data['uniform_points_inside'], k=4)
    dist_uniform_points_outside, idx_uniform_points_outside = kd_tree.query(pts_data['uniform_points_outside'], k=4)
    mat_fname = os.path.join(output_fd, 'sample/sample2smpl.mat')
    sio.savemat(mat_fname,
                {
                    'dist_surface_points_inside': dist_surface_points_inside,
                    'idx_surface_points_inside': idx_surface_points_inside,
                    'dist_surface_points_outside': dist_surface_points_outside,
                    'idx_surface_points_outside': idx_surface_points_outside,
                    'dist_uniform_points_inside': dist_uniform_points_inside,
                    'idx_uniform_points_inside': idx_uniform_points_inside,
                    'dist_uniform_points_outside': dist_uniform_points_outside,
                    'idx_uniform_points_outside': idx_uniform_points_outside,
                },
                do_compression=True)

    # # [for debugging]
    # with open('debug.obj', 'w') as fp:
    #     for p in surface_points_inside:
    #         fp.write('v %f %f %f\n' % (p[0], p[1], p[2]))
    #     for p in uniform_points_inside:
    #         fp.write('v %f %f %f\n' % (p[0], p[1], p[2]))


def main(worker_num=12):
    data_list = get_data_list()
    print('Found %d data items' % len(data_list))

    pool = multiprocessing.Pool(processes=worker_num)
    try:
        r = [pool.apply_async(process_one_data_item, args=(data_item,))
             for data_item in data_list]
        pool.close()
        for item in r:
            item.wait(timeout=9999999)
    except KeyboardInterrupt:
        pool.terminate()
    finally:
        pool.join()
        print('Done. ')

    # for data_item in tqdm(data_list, ascii=True):
    #     process_one_data_item(data_item)
    # print('Done')


if __name__ == '__main__':
    main()