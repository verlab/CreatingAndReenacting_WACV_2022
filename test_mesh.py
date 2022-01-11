import os
import sys
import argparse
from utils import dataloader
import torch
import torchvision
import cv2
from models.render import Render_SMPL,Render_TEX
from models.mesh import SMPL_Mesh,TEX_Mesh
from models.smpl import SMPL,load_smpl
from models.meshNet import MeshRefinementStage, MeshRefinementHead
from models.textureNet import TextureRefinementStage,discriminator
from utils.mesh_tools import write_obj
from utils.SSIM import SSIM
from utils import arguments

import random

from pytorch3d.io import save_obj

from pytorch3d.utils import ico_sphere
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)
from pytorch3d.structures import Meshes
from pytorch3d.transforms import Transform3d 

from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
from torch.autograd import Variable

import datetime
import yaml

import numpy as np
from tqdm import tqdm
import config

import pdb


# Losses to smooth / regularize the mesh shape
def update_mesh_shape_prior_losses(mesh,loss):
    # and (b) the edge length of the predicted mesh
    #loss["edge"] = mesh_edge_loss(mesh)

    edges_packed = mesh.edges_packed() 
    verts_packed = mesh.verts_packed()

    verts_edges = verts_packed[edges_packed]
    v0, v1 = verts_edges.unbind(1)

    edge_size = ((v0 - v1).norm(dim=1, p=2)) - 0.1
    m = torch.nn.ReLU()
    loss["edge"] = (m(edge_size)).sum()
 
    #loss["edge"] = mesh_edge_loss(mesh)

    # mesh normal consistency
    loss["normal"] = mesh_normal_consistency(mesh)
    
    # mesh laplacian smoothing
    loss["laplacian"] = mesh_laplacian_smoothing(mesh, method="uniform")

def draw_weights(model_name, model, summary, epoch):

    if model_name == 'texture':
        for i, (param_name, param) in enumerate(model.named_parameters()):
            summary.add_histogram(f"{model_name}/channel_0", param[..., 0].flatten().data.cpu(), epoch)
            summary.add_histogram(f"{model_name}/channel_1", param[..., 1].flatten().data.cpu(), epoch)
            summary.add_histogram(f"{model_name}/channel_2", param[..., 2].flatten().data.cpu(), epoch)

    if model_name == 'mesh':
        for i, (param_name, param) in enumerate(model.named_parameters()):
            try:
                _, stage, _, gconv, weight, weight_type = tuple(param_name.split('.'))
                summary.add_histogram(f"{model_name}/stage{stage}/gconv_{gconv}_{weight}.{weight_type}", param.data.cpu(), epoch)

            except ValueError:
                _, stage, _, weight_type = tuple(param_name.split('.'))
                summary.add_histogram(f"{model_name}/stage{stage}/verts_offset.{weight_type}", param.data.cpu(), epoch)

def save_model(state_dict, path):        
    torch.save(state_dict, path)
def read_model(path, device):
    return torch.load(path, map_location = device)



#python trainMesh.py -d /media/thiagoluange/SAMSUNG/ -w 1 -rss 256 -rsh 75 -b 1

#python trainMesh.py -d /media/thiagoluange/SAMSUNG/AIST/ -w 1 -rss 256 -rsh 512 -b 1 -st aist -g male -s d -p 04

def main():

    ## ARGS
    args = arguments.get_args()

    if(torch.cuda.is_available()):
        device = torch.device("cuda:{}".format(args.device))
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    print(f"RUNNING ON {device}")
    
    ## CREATE DATALOADER
    #dataloaders = dataloader.get_dataloaders(args)
    movements = ["box", "cone", "fusion", "hand", "jump", "rotate", "shake_hands", "simple_walk"]
    dataloaders, dataset = dataloader.get_dataloaders(args, phase = "test", movements = movements, test = True)
    #dataloader_test = dataloader.get_dataloaders(args, phase = "test")
    ## RECOVERING FIXED PARAMS
    #dataset   = dataloaders['test'].dataset

        
    faces_mesh = torch.from_numpy(np.load(config.SMPL_FACES)).to(device)
    output_path = os.path.join(args.output_path, args.source, args.person, "mesh_results")
    os.makedirs(output_path, exist_ok = True)
  
    f = dataset.f
    '''
        NOTE: Atualmente img_shape recebe o tamanho da imagem original. 
        Talvez tenha que mudar para tamanho da imagem cropada.
        Seria pegar a menor dimensao para montar imagem quadrada? Thiago: Deve receber o tamanho original mesmo, ele que defini a camera
    '''
    img_shape = dataset.img_shape

    ## LOAD TEXTURE
    txt_img = cv2.resize(cv2.imread(config.TEX_MAP, cv2.IMREAD_UNCHANGED), (512,512))
    '''
        NOTE: Nao sei o quanto o tamanho da textura vai influenciar na qualidade e no peso da rede
    '''

    txt_img =  torch.from_numpy(txt_img[:,:,::-1].astype('float64')/255.0).to(device)

    ### LOADING MODELS
    ## LOAD MESH MODEL
    with open("models/model_cfg.yaml", 'r') as cfg_file:
        model_cfgs = yaml.safe_load(cfg_file)

    model_cfgs["device"] = device
    model_cfgs["batch_size"] = args.batch_size


    model = MeshRefinementHead(model_cfgs).to(device)
    if(args.pretrained_path_model is not None):
        model.load_state_dict(read_model(args.pretrained_path_model, device))
        print("loaded weights sucessfully")

    ## LOAD RENDER MODEL

    if args.style == "mt":
        my_render_hard = Render_SMPL(f, img_shape, args.render_size_hard, device, "hard").to(device)  

    else:
        my_render_hard = Render_SMPL(f, img_shape, args.render_size_hard, device, "hard", eye=[[0,0,0]],at=[[0,0,-1]], up=[[0, 1, 0]]).to(device)  
    


    ## MODELS OPTIMIZER

    ## MODELS TRAIN
    model.eval()
    
    #cv2.imwrite(checkpoint_path + "/image_step_%09d_in"%step + ".jpg", img_hard.permute(0, 2, 3, 1).cpu().detach().numpy()[0,face_posi[1] - 40:face_posi[1] + 40,face_posi[0] - 40:face_posi[0] + 40,-1::-1]*255)
    step = 0
    for idx, (vertices, trans, global_mat, f_now) in enumerate(tqdm(dataloaders['test'])):

                    ############            Initialize optimizer mesh    
        trans = trans.to(device)

        ## CREATE MESH
        vertices = [vert.to(device) for vert in vertices]
        faces = [faces_mesh.to(device) for i in range(len(vertices))] ## O numero de amostras no batch sempre sera batch_size?
        src_mesh = Meshes(verts=vertices, faces=faces).to(device)
        
        with torch.no_grad():            
            # Deform the mesh
            subdivide = False

            deformed_mesh = model(src_mesh, subdivide)

            it_size = int(deformed_mesh.verts_packed().shape[0]/len(vertices))
            deformed_meshes = [deformed_mesh.verts_packed()[idx : idx + it_size] for idx in range(0, len(deformed_mesh.verts_packed()), it_size)]       

            tex_maps = []
            
            for i in range(len(vertices)):
                tex_maps.append(txt_img)

            tex_map = torch.stack(tex_maps)

            render_mesh = SMPL_Mesh(deformed_meshes, faces, tex_map, device)

            S = torch.ones(f_now.shape[0],3)

            for i in range(f_now.shape[0]):
                S[i,2] = f/f_now[i]    

            images_predicted = my_render_hard(render_mesh.to(device), trans,global_mat,S)
            predicted_silhouette = images_predicted[..., 3:].to(device)
            rgb = images_predicted[..., :3]
            seg = images_predicted[..., 3:]
            for idx in range(images_predicted.shape[0]):
                rgb_i = rgb.cpu().detach().numpy()[idx,:,:,-1::-1]*255
                seg_i = seg.cpu().detach().numpy()[idx,:,:,-1::-1]*255                
                cv2.imwrite(os.path.join(output_path, "TEST_mesh{:05d}.jpg").format(step), rgb_i)
                cv2.imwrite(os.path.join(output_path, "TEST_seg{:05d}.jpg").format(step), seg_i)
                step += 1

    os.system('ffmpeg -hide_banner -loglevel panic -framerate 30 -i {}/TEST_mesh%05d.jpg {}/{}.mp4'.format(output_path, output_path, "result_mesh"))
    os.system('ffmpeg -hide_banner -loglevel panic -framerate 30 -i {}/TEST_seg%05d.jpg {}/{}.mp4'.format(output_path, output_path, "result_seg"))

            #cv2.imwrite(checkpoint_path + "/image_step_%09d_in"%step + ".jpg", img_soft.permute(0, 2, 3, 1).cpu().detach().numpy()[0,:,:,-1::-1]*255)

            #cv2.imwrite(checkpoint_path + "/image_step_%09d_in"%step + ".jpg", img_hard.permute(0, 2, 3, 1).cpu().detach().numpy()[0,face_posi[1] - 20:face_posi[1] + 20,face_posi[0] - 20:face_posi[0] + 20,-1::-1]*255)
            #cv2.imwrite(checkpoint_path + "/seg_step_%09d_in"%step + ".png",  seg_soft.permute(0, 2, 3, 1).detach().numpy()[0,:,:,:]*255)

            #cv2.imwrite(checkpoint_path + "/image_step_%09d_out"%step + ".jpg", images_predicted[..., :3].cpu().detach().numpy()[0,:,:,-1::-1]*255)
            #cv2.imwrite(checkpoint_path + "/seg_step_%09d_out"%step + ".png",  images_predicted[..., 3:].detach().numpy()[0,:,:,:]*255)

            #out_vert,out_faces = render_mesh.get_mesh_verts_faces(0)

            #save_obj(checkpoint_path + "/model_%09d"%step + ".obj",out_vert,out_faces)

            

            # Optimization step
                


            ## TEST PHASE
            ## VOLTAR IDENTACAO
   
if __name__ == "__main__":
    main()
