import os
import torch
from skimage.io import imread
import cv2

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

import config
import constants
import numpy as np


def Template_tex():
        vt = np.load(config.SMPL_VT)
        ft = np.load(config.SMPL_FT)
        return vt,ft

def SMPL_Mesh(verts,faces,image,device):

	vt,ft = Template_tex()           
    
	verts_uvs = torch.from_numpy(vt)  # (V, 2)
	faces_uvs = torch.from_numpy(ft).to(device)  # (F, 3)
	verts_uvs = verts_uvs.type(torch.FloatTensor).to(device)
	image = image.type(torch.FloatTensor).to(device) 
	tex = TexturesUV(verts_uvs=[verts_uvs for i in range(len(verts))], faces_uvs=[faces_uvs for i in range(len(verts))], maps= image)

	mesh = Meshes(verts=verts, faces=faces, textures=tex)

	return mesh



def TEX_Mesh(normals,device):

	vt,ft = Template_tex()
	normal_z = normals.cpu().numpy()[:,2]

	vis_faces = np.where(normal_z > 0.0)
	ft = np.delete(ft,vis_faces,axis=0) 
   
    
	verts_uvs = torch.cat([torch.from_numpy(vt),torch.from_numpy(np.ones((vt.shape[0],1)))],dim=1)      
	verts_uvs = verts_uvs.type(torch.FloatTensor).to(device)	

	faces_uvs = torch.from_numpy(ft).to(device)  # (F, 3)

	mesh = Meshes(verts=[verts_uvs], faces=[faces_uvs])

	return mesh

