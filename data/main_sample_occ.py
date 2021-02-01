import numpy as np
import os
import cv2 as cv
import glob
import math
import random
from tqdm import tqdm
import scipy.io as sio
import trimesh
import trimesh.sample
import trimesh.curvature
import multiprocessing


""" 
runtime configuration 
"""
mesh_data_dir = '../dataset_example/mesh_data'
output_data_dir = '../dataset_example/image_data'
view_num = 360
cam_f = 5000
cam_dist = 10
img_res = 512
num_sample_surface = 400000
num_sample_uniform = 25000
sigma = 0.025
sigma_small = 0.01
curv_thresh = 0.004


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
    _, item_name = os.path.split(data_item[:-1])
    output_fd = os.path.join(output_data_dir, item_name)
    os.makedirs(output_fd, exist_ok=True)
    os.makedirs(os.path.join(output_fd, 'sample'), exist_ok=True)

    mesh = read_data(data_item)
    mesh_bbox_min = np.min(mesh.vertices, axis=0, keepdims=True)
    mesh_bbox_max = np.max(mesh.vertices, axis=0, keepdims=True)
    mesh_bbox_size = mesh_bbox_max - mesh_bbox_min

    surface_points, _ = trimesh.sample.sample_surface(mesh, num_sample_surface)
    curvs = trimesh.curvature.discrete_gaussian_curvature_measure(mesh, surface_points, 0.004)
    curvs = abs(curvs)
    curvs = curvs / max(curvs)  # normalize curvature
    sigmas = np.zeros(curvs.shape)
    sigmas[curvs <= curv_thresh] = sigma
    sigmas[curvs > curv_thresh] = sigma_small
    random_shifts = np.random.randn(surface_points.shape[0], surface_points.shape[1])
    random_shifts *= np.expand_dims(sigmas, axis=-1)
    surface_points = surface_points + random_shifts
    inside = mesh.contains(surface_points)
    surface_points_inside = surface_points[inside]
    surface_points_outside = surface_points[np.logical_not(inside)]

    uniform_points1 = np.random.rand(num_sample_uniform * 2, 3) * mesh_bbox_size + mesh_bbox_min
    uniform_points2 = np.random.rand(num_sample_uniform, 3) * 1.0 - 0.5
    inside1 = mesh.contains(uniform_points1)
    inside2 = mesh.contains(uniform_points2)
    uniform_points_inside = uniform_points1[inside1]
    uniform_points_outside = uniform_points2[np.logical_not(inside2)]
    if len(uniform_points_inside) > num_sample_uniform // 2:
        uniform_points_inside = uniform_points_inside[:(num_sample_uniform // 2)]
        uniform_points_outside = uniform_points_outside[:(num_sample_uniform // 2)]
    else:
        uniform_points_outside = uniform_points_outside[:(num_sample_uniform - len(uniform_points_inside))]

    sio.savemat(os.path.join(output_fd, 'sample', 'samples.mat'),
                {
                    'surface_points_inside': surface_points_inside,
                    'surface_points_outside': surface_points_outside,
                    'uniform_points_inside': uniform_points_inside,
                    'uniform_points_outside': uniform_points_outside
                }, do_compression=True)
    sio.savemat(os.path.join(output_fd, 'sample', 'meta.mat'),
                {
                    'sigma': sigma,
                    'sigma_small': sigma_small,
                    'curv_thresh': curv_thresh,
                })

    # # # [for debugging]
    # print(len(uniform_points_inside))
    # print(len(uniform_points_outside))
    # with open('debug.obj', 'w') as fp:
    #     for p in uniform_points_inside:
    #         fp.write('v %f %f %f 0 1 0\n' % (p[0], p[1], p[2]))
    #     for p in uniform_points_outside:
    #         fp.write('v %f %f %f 1 0 0\n' % (p[0], p[1], p[2]))
    # import pdb
    # pdb.set_trace()


def main(worker_num=8):
    data_list = get_data_list()
    print('Found %d data items' % len(data_list))

    # for data_item in tqdm(data_list, ascii=True):
    #     process_one_data_item(data_item)
    # print('Done')
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


if __name__ == '__main__':
    main()