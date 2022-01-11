import os
import torch
from skimage.io import imread
import cv2

import torch.nn as nn
import numpy as np
import pdb

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    FoVOrthographicCameras,
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesUV,
    BlendParams,
    SoftSilhouetteShader,
)

from pytorch3d.transforms import Transform3d 

def mim(x1,x2):
        if x1 < x2:
                return x1
        else:
                return x2

def load_render(f,img_size,render_size,device, eye=[[0,0,0]],at=[[0,0,1]], up=[[0, -1, 0]], sigma=1e-5, faces_per_pixel = 30):

        R, T = look_at_view_transform(eye=eye,at=at, up=up)
        #R, T = look_at_view_transform(dist=2.7, elev=10, azim=-150)

        tan_theta = (min(img_size[0],img_size[1])/2.0)/f

        abertura = ((np.arctan(tan_theta)*2)*180)/np.pi

        cameras = FoVPerspectiveCameras(fov=abertura,device=device, R=R, T=T)
        
        
        #raster_settings_soft = RasterizationSettings(image_size=128,blur_radius=np.log(1. / 1e-4 - 1.)*sigma,faces_per_pixel=50)

        #raster_settings = RasterizationSettings(image_size=512,blur_radius=0.0,faces_per_pixel=1)

        raster_settings = RasterizationSettings(image_size=render_size,blur_radius=np.log(1. / 1e-4 - 1.)*sigma,faces_per_pixel=faces_per_pixel)

        lights = PointLights(device=device,ambient_color= [[1.0, 1.0, 1.0]], diffuse_color=[[0.0, 0.0, 0.0]], specular_color=[[0.0,0.0,0.0]],location=[[0.0, 0.0, 0.0]])
        
        renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                        cameras=cameras, 
                        raster_settings=raster_settings
                ),
                shader=SoftPhongShader(
                        device=device, 
                        cameras=cameras,
                        lights=lights
                )
        )

        return renderer



class Render_SMPL(nn.Module):

        def __init__(self, f,img_size,render_size,device,render_type = "soft",eye=[[0,0,0]],at=[[0,0,1]], up=[[0, -1, 0]]):
                super(Render_SMPL, self).__init__()
                self.render_type = render_type
                self.render = load_render(f,img_size,render_size,device,eye=eye,at=at, up=up) if self.render_type == "soft" else load_render(f,img_size,render_size,device, eye=eye,at=at, up=up,sigma = 0, faces_per_pixel = 1)
                self.device = device

        def forward(self,mesh,trans,global_mat,S):
                
                
                #mesh_trans = torch.empty(0, 3).to(self.device) 
                #for i in range(trans.shape[0]):
                #     mesh_trans = torch.cat([mesh_trans, trans[i,:].expand(int(mesh.verts_packed().shape[0]/trans.shape[0]),mesh.verts_packed().shape[1])], dim=0)

                # this work only to batch 1

                my_transform = Transform3d(device=self.device, matrix=torch.transpose(global_mat.view(-1, 4, 4).to(self.device), 1, 2)).translate(trans[:,0],trans[:,1], trans[:,2]).scale(S)
  
                verts_camera = my_transform.transform_points(mesh.verts_padded())

                mesh2render = mesh.update_padded(new_verts_padded=verts_camera)

                #mesh.offset_verts_(mesh_trans)
           
                images = self.render(mesh2render.to(self.device))

                return images


def load_render_tex(render_size,device,sigma=0.0, faces_per_pixel = 1):

        cameras = FoVOrthographicCameras(znear=1.0, zfar=100.0, max_y=1.0, min_y=0.0, max_x=0.0, min_x=1.0, device=device)        

        raster_settings = RasterizationSettings(image_size=render_size,blur_radius=np.log(1. / 1e-4 - 1.)*sigma,faces_per_pixel=faces_per_pixel)
        
        blend_params = BlendParams(sigma=0.0, gamma=0.0)

        renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                        cameras=cameras, 
                        raster_settings=raster_settings
                ),
                shader=SoftSilhouetteShader(blend_params=blend_params)
        )

        return renderer



class Render_TEX(nn.Module):

        def __init__(self,render_size,device):
                super(Render_TEX, self).__init__()
                self.render = load_render_tex(render_size,device)
                self.device = device

        def forward(self,mesh):

                images = self.render(mesh)

                return images




