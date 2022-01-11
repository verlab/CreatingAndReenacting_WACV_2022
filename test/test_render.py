import os
import torch
from skimage.io import imread
import cv2
#python test_render.py --model_file /media/thiagoluange/SAMSUNG/mt-dataset/S1/box/P0/test_pose_new/GOPR95380000000001.jpg_body.pkl --model_type 2 --model_texture /media/thiagoluange/SAMSUNG/mt-dataset/S1/A-pose/P0/video-avatar/tex-P0.jpg
# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesUV
)

import sys
sys.path.append("..")

from models.render import Render_SMPL
from models.mesh import SMPL_Mesh
from models.smpl import SMPL,load_smpl
from utils.mesh_tools import write_obj

import numpy as np
import pickle
import smplx
import config
import constants
import pdb
import argparse


parser = argparse.ArgumentParser(description='Process train arguments.')
parser.add_argument('--model_file', required=True, help='Path to coarse model')
parser.add_argument('--model_type', type=int, default=0, help='(0) neutral, (1) male and (2) female')
parser.add_argument('--model_texture', required=True, help='Path to coarse texture file')

if __name__ == '__main__':
    
	args = parser.parse_args()

	my_smpl,betas = load_smpl(args.model_file,args.model_type)
        
	with open(args.model_file,'rb') as file_model:
		avatar = pickle.load(file_model,encoding='latin1')

	image = cv2.imread(args.model_texture)


	global_orient= torch.from_numpy(avatar['pose'][:3].reshape((1,-1)))     
	body_pose= torch.from_numpy(avatar['pose'][3:].reshape((1,-1))) 

	betas = torch.from_numpy(betas)
   
	smpl_output = my_smpl(global_orient=global_orient,body_pose=body_pose,betas=betas)

	if torch.cuda.is_available():
		device = torch.device("cuda:0")
		torch.cuda.set_device(device)
	else:
		device = torch.device("cpu")

	my_mesh = SMPL_Mesh(smpl_output.vertices.view(-1,3), torch.from_numpy((my_smpl.faces).astype('int32')),torch.from_numpy(image[:,:,::-1].astype('float64')/255.0),device)
        
	my_render = Render_SMPL(avatar['f'],avatar['img_shape'],512,device)
	
	image = my_render(my_mesh,np.array(avatar['trans']))
       
	cv2.imwrite("teste.jpg", image[0, ..., :3].cpu().numpy()[:,:,::-1]*255)

	#print (data_structure.keys())



