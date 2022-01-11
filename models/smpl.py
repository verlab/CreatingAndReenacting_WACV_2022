import torch
import numpy as np
import smplx
from smplx import SMPL as _SMPL
from smplx.body_models import ModelOutput
from smplx.lbs import vertices2joints

import config
import constants
import pickle
import os
import os.path as osp

import pdb



class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


def load_smpl(file_model_pkl,model_type):

	with open(file_model_pkl,'rb') as file_model:
		model_refined = pickle.load(file_model,encoding='latin1')

	if model_type == 1:
		gender="male"
	elif model_type == 2: 
		gender="female"
	else: 
		gender="neutral"


	model_path = config.SMPL_MODEL_DIR

	if osp.isdir(model_path):
		model_fn = 'SMPL_{}.{ext}'.format(gender.upper(), ext='pkl')
		smpl_path = os.path.join(model_path, model_fn)
	else:
		smpl_path = model_path
		assert osp.exists(smpl_path), 'Path {} does not exist!'.format(smpl_path)

	with open(smpl_path, 'rb') as smpl_file:

		model = pickle.load(smpl_file,encoding='latin1')

		if 'v_personal' in model_refined.keys():
			model['v_template'] = model['v_template'] + model_refined['v_personal']

		data_struct = Struct(**model)


	
	

	my_smpl = SMPL("",data_struct=data_struct ,batch_size=1,create_transl=False,gender=gender)
 
	return my_smpl, np.reshape(model_refined['betas'],(1,10)).astype(np.float32)


class SMPL(_SMPL):
    """ Extension of the official SMPL implementation to support more joints """

    def __init__(self, *args, **kwargs):
        super(SMPL, self).__init__(*args, **kwargs)
        joints = [constants.JOINT_MAP[i] for i in constants.JOINT_NAMES]
        J_regressor_extra = np.load(config.JOINT_REGRESSOR_TRAIN_EXTRA)
        self.register_buffer('J_regressor_extra', torch.tensor(J_regressor_extra, dtype=torch.float32))
        self.joint_map = torch.tensor(joints, dtype=torch.long)

    def forward(self, *args, **kwargs): #global_orient=global_orient,body_pose=body_pose,betas=betas
        kwargs['get_skin'] = True
        smpl_output = super(SMPL, self).forward(*args, **kwargs)
        extra_joints = vertices2joints(self.J_regressor_extra, smpl_output.vertices)
        joints = torch.cat([smpl_output.joints, extra_joints], dim=1)
        #joints = joints[:, self.joint_map, :]
        output = ModelOutput(vertices=smpl_output.vertices,
                             global_orient=smpl_output.global_orient,
                             body_pose=smpl_output.body_pose,
                             joints=joints,
                             betas=smpl_output.betas,
                             full_pose=smpl_output.full_pose)
        return output
