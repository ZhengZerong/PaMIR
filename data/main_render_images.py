import numpy as np
import os
import cv2 as cv
import glob
import math
import random
import pyexr
from tqdm import tqdm
import scipy.io as sio
import prt.sh_util as sh_util

from renderer.camera import Camera
from renderer.mesh import load_obj_mesh, compute_tangent, compute_normal, load_obj_mesh_mtl
from renderer.camera import Camera


""" 
runtime configuration 
"""
mesh_data_dir = '../dataset_example/mesh_data'
output_data_dir = '../dataset_example/image_data'
view_num = 360
cam_f = 5000
cam_dist = 10
img_res = 512


def get_data_list():
    """reads data list"""
    data_list = glob.glob(os.path.join(mesh_data_dir, './*/'))
    return sorted(data_list)


def read_data(item):
    """reads data """
    mesh_filename = glob.glob(os.path.join(item, '*.obj'))[0]  # assumes one .obj file
    text_filename = glob.glob(os.path.join(item, '*.jpg'))[0]  # assumes one .jpg file

    vertices, faces, normals, faces_normals, textures, face_textures \
        = load_obj_mesh(mesh_filename, with_normal=True, with_texture=True)
    texture_image = cv.imread(text_filename)
    texture_image = cv.cvtColor(texture_image, cv.COLOR_BGR2RGB)

    prt_data = sio.loadmat(os.path.join(item, 'bounce/prt_data.mat'))
    prt, face_prt = prt_data['bounce0'], prt_data['face']
    return vertices, faces, normals, faces_normals, textures, face_textures, texture_image, prt, face_prt


def make_rotate(rx, ry, rz):
    sinX = np.sin(rx)
    sinY = np.sin(ry)
    sinZ = np.sin(rz)

    cosX = np.cos(rx)
    cosY = np.cos(ry)
    cosZ = np.cos(rz)

    Rx = np.zeros((3,3))
    Rx[0, 0] = 1.0
    Rx[1, 1] = cosX
    Rx[1, 2] = -sinX
    Rx[2, 1] = sinX
    Rx[2, 2] = cosX

    Ry = np.zeros((3,3))
    Ry[0, 0] = cosY
    Ry[0, 2] = sinY
    Ry[1, 1] = 1.0
    Ry[2, 0] = -sinY
    Ry[2, 2] = cosY

    Rz = np.zeros((3,3))
    Rz[0, 0] = cosZ
    Rz[0, 1] = -sinZ
    Rz[1, 0] = sinZ
    Rz[1, 1] = cosZ
    Rz[2, 2] = 1.0

    R = np.matmul(np.matmul(Rz,Ry),Rx)
    return R


def generate_cameras(dist=10, view_num=60):
    cams = []
    target = [0, 0, 0]
    up = [0, 1, 0]
    for view_idx in range(view_num):
        angle = (math.pi * 2 / view_num) * view_idx
        eye = np.asarray([dist * math.sin(angle), 0, dist * math.cos(angle)])

        fwd = np.asarray(target, np.float64) - eye
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, up)
        right /= np.linalg.norm(right)
        down = np.cross(fwd, right)

        cams.append(
            {
                'center': eye, 
                'direction': fwd, 
                'right': right, 
                'up': -down, 
            }
        )

    return cams


def process_one_data_item(data_item, rndr, rndr_uv, shs):
    _, item_name = os.path.split(data_item[:-1])
    output_fd = os.path.join(output_data_dir, item_name)
    os.makedirs(output_fd, exist_ok=True)
    os.makedirs(os.path.join(output_fd, 'color'), exist_ok=True)
    os.makedirs(os.path.join(output_fd, 'mask'), exist_ok=True)
    os.makedirs(os.path.join(output_fd, 'color_uv'), exist_ok=True)
    os.makedirs(os.path.join(output_fd, 'meta'), exist_ok=True)

    vertices, faces, normals, faces_normals, textures, face_textures, \
        texture_image, prt, face_prt = read_data(data_item)

    cam = Camera(width=img_res, height=img_res, focal=5000, near=0.1, far=40)
    cam.sanity_check()

    rndr.set_norm_mat(1.0, 0.0)
    tan, bitan = compute_tangent(vertices, faces, normals, textures, face_textures)
    rndr.set_mesh(vertices, faces, normals, faces_normals, textures, face_textures, prt, face_prt, tan, bitan)    
    rndr.set_albedo(texture_image)
    rndr_uv.set_mesh(vertices, faces, normals, faces_normals, textures, face_textures, prt, face_prt, tan, bitan)    
    rndr_uv.set_albedo(texture_image)

    cam_params = generate_cameras(dist=cam_dist, view_num=view_num)
    sh_list = []
    for ci, cam_param in enumerate(tqdm(cam_params, ascii=True)):
        cam.center = cam_param['center']
        cam.right = cam_param['right']
        cam.up = cam_param['up']
        cam.direction = cam_param['direction']
        cam.sanity_check()
        rndr.set_camera(cam)
        rndr_uv.set_camera(cam)

        sh_id = random.randint(0,shs.shape[0]-1)
        sh = shs[sh_id]
        sh_angle = 0.2*np.pi*(random.random()-0.5)
        sh = sh_util.rotateSH(sh, make_rotate(0, sh_angle, 0).T)
        sh_list.append(sh)

        rndr.set_sh(sh)        
        rndr.analytic = False
        rndr.use_inverse_depth = False
        rndr.display()

        out_all_f = rndr.get_color(0)
        out_mask = out_all_f[:,:,3]
        out_all_f = cv.cvtColor(out_all_f, cv.COLOR_RGBA2BGR)

        cv.imwrite(os.path.join(output_fd, 'color', '%04d.jpg' % ci), np.uint8(out_all_f * 255))
        cv.imwrite(os.path.join(output_fd, 'mask', '%04d.png' % ci), np.uint8(out_mask * 255))

        rndr_uv.set_sh(sh)
        rndr_uv.analytic = False
        rndr_uv.use_inverse_depth = False
        rndr_uv.display()

        uv_color = rndr_uv.get_color(0)
        uv_color = cv.cvtColor(uv_color, cv.COLOR_RGBA2BGR)
        cv.imwrite(os.path.join(output_fd, 'color_uv', '%04d.png' % ci), np.uint8(uv_color * 255))

        if ci == 0:
            uv_pos = rndr_uv.get_color(1)
            uv_mask = uv_pos[:,:,3]
            # cv2.imwrite(os.path.join(out_path, 'UV_MASK', subject_name, '00.png'),255.0*uv_mask)
            cv.imwrite(os.path.join(output_fd, 'meta', 'uv_mask.png'), np.uint8(uv_mask * 255))

            data = {'default': uv_pos[:,:,:3]} # default is a reserved name
            pyexr.write(os.path.join(output_fd, 'meta', 'uv_pos.exr'), data)
            # cv.imwrite(os.path.join(output_fd, 'meta' '%04d.png' % ci), np.uint8(uv_color * 255))

            uv_nml = rndr_uv.get_color(2)
            uv_nml = cv.cvtColor(uv_nml, cv.COLOR_RGBA2BGR)
            # cv2.imwrite(os.path.join(out_path, 'UV_NORMAL', subject_name, '00.png'),255.0*uv_nml)
            cv.imwrite(os.path.join(output_fd, 'meta', 'uv_nml.png'), np.uint8(uv_nml * 255))

    sio.savemat(
        os.path.join(output_fd, 'meta', 'cam_data.mat'),
        {'cam': cam_params})
    sio.savemat(
        os.path.join(output_fd, 'meta', 'sh_data.mat'), 
        {'sh': sh_list})

    # import pdb
    # pdb.set_trace()


def main():
    shs = np.load('./env_sh.npy')
    egl = False
    
    from renderer.gl.init_gl import initialize_GL_context
    initialize_GL_context(width=img_res, height=img_res, egl=egl)

    from renderer.gl.prt_render import PRTRender
    rndr = PRTRender(width=img_res, height=img_res, ms_rate=1.0, egl=egl)
    rndr_uv = PRTRender(width=img_res, height=img_res, uv_mode=True, egl=egl)

    data_list = get_data_list()
    for data_item in tqdm(data_list, ascii=True):
        process_one_data_item(data_item, rndr, rndr_uv, shs)
    print('Done')


if __name__ == '__main__':
    main()