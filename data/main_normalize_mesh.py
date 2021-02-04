import numpy as np
import os
import glob
import multiprocessing
import tqdm
import trimesh

import objio
import prt.prt_util as prt_util

mesh_dir = '../dataset_example/mesh_data'


def get_data_list():
    """reads data list"""
    data_list = glob.glob(os.path.join(mesh_dir, './*/'))
    return sorted(data_list)


def get_mesh_tex_fname(folder):
    obj_list = glob.glob(os.path.join(folder, '*.obj'))
    jpg_list = glob.glob(os.path.join(folder, '*.jpg'))
    assert len(obj_list)==1 and len(jpg_list)==1, '[ERROR] More than one obj/jpg file are found!'
    return obj_list[0], jpg_list[0]


def process_one_data_item(data_item):
    _, item_name = os.path.split(data_item[:-1])
    source_fd = os.path.join(mesh_dir, item_name)
    obj_fname, tex_fname = get_mesh_tex_fname(source_fd)
    mesh = trimesh.load(obj_fname)
    mesh_v = mesh.vertices
    min_xyz = np.min(mesh_v, axis=0, keepdims=True)
    max_xyz = np.max(mesh_v, axis=0, keepdims=True)
    mesh_v = mesh_v - (min_xyz + max_xyz) * 0.5

    scale_inv = np.max(max_xyz-min_xyz)
    scale = 1.0 / scale_inv * (0.75 + np.random.rand() * 0.15)
    mesh_v *= scale
    mesh.vertices = mesh_v
    trimesh.base.export_mesh(mesh, obj_fname)


def main(worker_num=4):
    os.makedirs(mesh_dir, exist_ok=True)

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
    # for data_item in tqdm.tqdm(data_list, ascii=True):
    #     process_one_data_item(data_item)
    #     import pdb
    #     pdb.set_trace()
    print('Done')


if __name__ == '__main__':
    main()