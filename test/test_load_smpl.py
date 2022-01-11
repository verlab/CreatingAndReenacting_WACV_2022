import torch
import numpy as np
import pickle
import smplx

import sys
sys.path.append("..")

from models.smpl import SMPL,load_smpl
from utils.mesh_tools import write_obj
import config
import constants
import pdb
import argparse

parser = argparse.ArgumentParser(description='Process train arguments.')
parser.add_argument('--model_file', required=True, help='Path to coarse model')
parser.add_argument('--model_type', type=int, default=0, help='(0) neutral, (1) male and (2) female')

if __name__ == '__main__':
    
	args = parser.parse_args()

	my_smpl,betas = load_smpl(args.model_file,args.model_type)

	global_orient=torch.zeros(1,3)
	body_pose=torch.zeros(1,69)
	betas = torch.from_numpy(betas)
   
	smpl_output = my_smpl(global_orient=global_orient,body_pose=body_pose,betas=betas)

	write_obj("mesh.obj",my_smpl.faces,smpl_output.vertices.view(-1,3).numpy())

       
	#print (data_structure.keys())
