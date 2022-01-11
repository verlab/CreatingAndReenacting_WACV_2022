"""
This file contains definitions of useful data stuctures and the paths
for the datasets and data files necessary to run the code.
"""
import os
this_path = os.path.dirname(os.path.abspath(__file__))
print (this_path)




CUBE_PARTS_FILE = '/code/data/cube_parts.npy'
JOINT_REGRESSOR_TRAIN_EXTRA = '/code/data/J_regressor_extra.npy'
JOINT_REGRESSOR_H36M = '/code/data/J_regressor_h36m.npy'
VERTEX_TEXTURE_FILE = '/code/data/vertex_texture.npy'
STATIC_FITS_DIR = '/code/data/static_fits'
SMPL_MEAN_PARAMS = '/code/data/smpl_mean_params.npz'
SMPL_MODEL_DIR = '/code/data/smpl'

SMPL_FACES = this_path + '/models/new_faces.npy'
SMPL_IN_VERT = this_path + '/models/in_vertices.txt'
SMPL_VT = this_path + '/models/new_vt.npy'
SMPL_FT = this_path + '/models/new_ft.npy'
ANCHOR_V = this_path + '/models/map_anchor_v_new.npz'
TEX_MAP = this_path + '/models/texture.jpg'

TEX_MAP = this_path + '/models/texture.jpg'

_MODEL_MAP_FNAME = this_path + '/models/map_14_partes.txt'


