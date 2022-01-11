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
from models.p2p_networks import discriminator_mesh
from utils import arguments

from utils.preprocess_noise import Dilation2d

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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import numpy as np
from tqdm import tqdm
import config

import pdb

import PIL.Image
from torchvision.transforms import ToTensor
import io

torch.backends.cudnn.benchmark = True



def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            if p.grad is None:
                ave_grads.append(0)
                max_grads.append(0)
                print("NONE!")
            else:
                ave_grads.append(p.grad.abs().mean())
                max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])

    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    return buf

def get_grad_img(model):
    plot_buf = plot_grad_flow(model.named_parameters())

    im_grads = PIL.Image.open(plot_buf)
    im_grads = ToTensor()(im_grads).unsqueeze(0)

    return im_grads


# Losses to smooth / regularize the mesh shape
def update_mesh_shape_prior_losses(src_mesh,mesh,loss,batch_size,device):
    # and (b) the edge length of the predicted mesh
    #loss["edge"] = mesh_edge_loss(mesh)

  

    '''edges_packed = mesh.edges_packed() 
    verts_packed = mesh.verts_packed()

    verts_edges = verts_packed[edges_packed]
    v0, v1 = verts_edges.unbind(1)

    edge_size = ((v0 - v1).norm(dim=1, p=2)) - 0.1
    m = torch.nn.ReLU()
    loss["edge"] = (m(edge_size)).sum()
 
    #loss["edge"] = mesh_edge_loss(mesh)'''

    

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



#python trainMesh.py -d /media/thiagoluange/SAMSUNG/ -w 1 -rss 128 -rsh 75 -b 1

#python trainMesh.py -d /media/thiagoluange/SAMSUNG/AIST/ -w 1 -rss 256 -rsh 512 -b 1 -st aist -g male -s d -p 04

def main():

    ## ARGS
    args = arguments.get_args()

    ## TEXTURE PATH: "/srv/storage/datasets/thiagoluange/dd_dataset/S1P0/tex.jpg"

    ## SUMMARY & CHECKPOINTS
    checkpoint_path = args.dataset_path + '/checkpoints_meshNet-iccv_1.1/'
    summary_dir = args.dataset_path + '/summaries_meshNet-iccv_1.1/'

    dataset_person = f"{args.source}{args.person}"

    date = datetime.datetime.now()        
    time_init    = f"{date.day}-{date.month}-{date.year}_{date.hour}:{date.minute}:{date.second}"
    if args.flag is not None:
        time_init = "{}_{}".format(time_init, args.flag)
    summary_path = f"{summary_dir}/{dataset_person}/{args.experiment_name}_meshNet_lr_{args.lr}-le_{args.loss_edge}-lss_{args.loss_ssim}-ls_{args.loss_sil}-ll_{args.loss_lap}-ln_{args.loss_nor}/batch_{args.batch_size}/epochs_{args.epochs}/{time_init}/"
    weights_path =   f"{checkpoint_path}/{dataset_person}/{args.experiment_name}_meshNet_lr_{args.lr}-le_{args.loss_edge}-lss_{args.loss_ssim}-ls_{args.loss_sil}-ll_{args.loss_lap}-ln_{args.loss_nor}/batch_{args.batch_size}/epochs_{args.epochs}/{time_init}/"
   
    os.makedirs(summary_path, exist_ok=True)
    os.makedirs(weights_path, exist_ok=True)

    summary = SummaryWriter(log_dir=summary_path)

    ## GETTING DEVICE
    if(torch.cuda.is_available()):
        device = torch.device("cuda:{}".format(args.device))
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    print(f"RUNNING ON {device}")
    
    ## CREATE DATALOADER
    #dataloaders = dataloader.get_dataloaders(args)
    dataloaders = dataloader.get_dataloaders(args, phase = "train")
    #dataloader_test = dataloader.get_dataloaders(args, phase = "test")
    ## RECOVERING FIXED PARAMS
    dataset   = dataloaders['train'].dataset

        
    faces_mesh = torch.from_numpy(np.load(config.SMPL_FACES)).to(device)

  
    f = dataset.f
    '''
        NOTE: Atualmente img_shape recebe o tamanho da imagem original. 
        Talvez tenha que mudar para tamanho da imagem cropada.
        Seria pegar a menor dimensao para montar imagem quadrada? Thiago: Deve receber o tamanho original mesmo, ele que defini a camera
    '''
    img_shape = dataset.img_shape

    ## LOAD TEXTURE
    txt_img = cv2.resize(cv2.imread(config.TEX_MAP, cv2.IMREAD_UNCHANGED),(256,256))
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

    '''if(args.pretrained_path_model is not None):
        model.load_state_dict(read_model(args.pretrained_path_model, device))
        print("loaded weights sucessfully")'''

    ## LOAD RENDER MODEL

    if args.style == "mt":
        my_render_soft = Render_SMPL(f, img_shape, args.render_size_soft, device).to(device)  

    else:
        my_render_soft = Render_SMPL(f, img_shape, args.render_size_soft, device,eye=[[0,0,0]],at=[[0,0,-1]], up=[[0, 1, 0]]).to(device)   

    ## MODELS OPTIMIZER
    optimizer_mesh = torch.optim.AdamW([
                                    {'params': model.parameters()}
                                 #   {'params': model_tex.parameters()}
                                 ], lr=args.lr, betas=(0.5, 0.999))
   

    #we keep the same learning rate for the first args.epochs/2
    #and linearly decay the rate to zero over the args.epochs/2
    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch - args.epochs/2) / float(args.epochs/2 + 1)
        return lr_l

    scheduler_mesh = lr_scheduler.LambdaLR(optimizer_mesh, lr_lambda=lambda_rule)

    ## SETTING LOSSES
    losses = {"silhouette": {"weight": args.loss_sil},
        "ssim": {"weight": args.loss_ssim},
        "edge": {"weight": args.loss_edge},
        "normal": {"weight": args.loss_nor},
        "laplacian": {"weight": args.loss_lap},
    }


    step = 0
    best_loss = sys.maxsize
    #dataset_test = list(dataloader_test["test"])

    '''model_tex_D = discriminator_mesh(1).to(device)
    model_tex_D.weight_init(mean=0.0, std=0.02)

    optimizer_tex_D = torch.optim.AdamW([
                                #{'params': model.parameters()}
                                {'params': model_tex_D.parameters()}
                                ], lr=args.lr_tex/args.lr_d_factor, betas=(0.5, 0.999))

    scheduler_tex_D = lr_scheduler.LambdaLR(optimizer_tex_D, lr_lambda=lambda_rule)

    model_tex_D.train()
    BCE_loss = torch.nn.MSELoss().to(device)'''
    

    ## MODELS TRAIN
    model.train()

    # create loss
    model_ssim = SSIM().to(device)
   
    
    #cv2.imwrite(checkpoint_path + "/image_step_%09d_in"%step + ".jpg", img_hard.permute(0, 2, 3, 1).cpu().detach().numpy()[0,face_posi[1] - 40:face_posi[1] + 40,face_posi[0] - 40:face_posi[0] + 40,-1::-1]*255)


    my_render_hard = Render_SMPL(f, img_shape, args.render_size_soft, device, "hard").to(device) 


    my_dilation = Dilation2d(1,1,args.dilate_kernel,device)



    for epoch in range(args.epochs):
        print(f"EPOCH: {epoch}/{args.epochs}") 
        for idx, (vertices, seg_soft,seg_hard,img_soft,img_hard,trans,global_mat,f_now,face_posi) in enumerate(tqdm(dataloaders['train'])):


                                    ############            Initialize optimizer mesh    
            optimizer_mesh.zero_grad()

            seg_soft = seg_soft.to(device) 

            seg_soft_d = my_dilation(seg_soft)           
            trans = trans.to(device)
            img_soft = img_soft.to(device)
           
            batch_size = vertices.shape[0]              

            ## CREATE MESH
            vertices = [vert.to(device) for vert in vertices]
            faces = [faces_mesh.to(device) for i in range(len(vertices))] ## O numero de amostras no batch sempre sera batch_size?
            src_mesh = Meshes(verts=vertices, faces=faces).to(device)
         
            
            # Deform the mesh
            subdivide = False

            deformed_mesh = model(src_mesh, subdivide)

            #pdb.set_trace()

            it_size = int(deformed_mesh.verts_packed().shape[0]/len(vertices))
            deformed_meshes = [deformed_mesh.verts_packed()[idx : idx + it_size] for idx in range(0, len(deformed_mesh.verts_packed()), it_size)]       
  
            tex_maps = []
            
            for i in range(len(vertices)):
                 tex_maps.append(txt_img)

            tex_map = torch.stack(tex_maps)

            render_mesh = SMPL_Mesh(deformed_meshes, faces, tex_map, device)
            

            #Losses to smooth /regularize the mesh shape
            loss = {k: torch.tensor(0.0, device=device) for k in losses}
            
            update_mesh_shape_prior_losses(src_mesh,deformed_mesh,loss,batch_size,device)

            S = torch.ones(f_now.shape[0],3)

            for i in range(f_now.shape[0]):
                 S[i,2] = f/f_now[i]    

            images_predicted = my_render_soft(render_mesh.to(device), trans,global_mat,S)

            num_views_per_iteration = img_soft.shape[0]

            predicted_silhouette = images_predicted[..., 3:].to(device)
            predicted_seg = (torch.where(predicted_silhouette < 0.001, predicted_silhouette, torch.ones_like(predicted_silhouette))) ## NEW LINE

            loss_silhouette =  torch.tensor(1.0, device=device) - torch.norm(predicted_silhouette*seg_soft_d.permute(0, 2, 3, 1),1)/torch.norm(predicted_silhouette + seg_soft_d.permute(0, 2, 3, 1) - predicted_silhouette*seg_soft_d.permute(0, 2, 3, 1),1)

            loss_ssim = 1.0 - model_ssim(seg_soft_d,predicted_silhouette.permute(0, 3, 1, 2))        

            loss["ssim"] +=  loss_ssim 

            loss["silhouette"] += loss_silhouette           
           

            # Weighted sum of the losses
            sum_loss = torch.tensor(0.0, device=device)

            for k, l in loss.items():
                sum_loss += l * losses[k]["weight"]
            # Print the losses
            #sum_loss = sum_loss
          
            #import pdb
            #pdb.set_trace()
          
            

            #cv2.imwrite(checkpoint_path + "/image_step_%09d_in"%step + ".jpg", img_soft.permute(0, 2, 3, 1).cpu().detach().numpy()[0,:,:,-1::-1]*255)

            #cv2.imwrite(checkpoint_path + "/image_step_%09d_in"%step + ".jpg", img_hard.permute(0, 2, 3, 1).cpu().detach().numpy()[0,face_posi[1] - 20:face_posi[1] + 20,face_posi[0] - 20:face_posi[0] + 20,-1::-1]*255)
            #cv2.imwrite(checkpoint_path + "/seg_step_%09d_in"%step + ".png",  seg_soft.permute(0, 2, 3, 1).detach().numpy()[0,:,:,:]*255)

            #cv2.imwrite(checkpoint_path + "/image_step_%09d_out"%step + ".jpg", images_predicted[..., :3].cpu().detach().numpy()[0,:,:,-1::-1]*255)
            #cv2.imwrite(checkpoint_path + "/seg_step_%09d_out"%step + ".png",  images_predicted[..., 3:].detach().numpy()[0,:,:,:]*255)

            

            

            #cv2.imwrite(checkpoint_path + "/seg_step_%09d_in_2"%step + ".png",  seg_soft.permute(0, 2, 3, 1).detach().numpy()[0,:,:,:]*255)

            #cv2.imwrite(checkpoint_path + "/image_step_%09d_out_2"%step + ".jpg", images_predicted[..., :3].cpu().detach().numpy()[0,:,:,-1::-1]*255)
            #cv2.imwrite(checkpoint_path + "/seg_step_%09d_out_2"%step + ".png",  images_predicted[..., 3:].detach().numpy()[0,:,:,:]*255)

            #out_vert,out_faces = render_mesh.get_mesh_verts_faces(0)

            #save_obj(checkpoint_path + "/model_%09d"%step + ".obj",out_vert,out_faces)

            # Optimization step
            sum_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 2)

            optimizer_mesh.step()

            if step%args.delta_test == 0:
                grad_img = get_grad_img(model)
        
            step = step + 1

            summary.add_scalar('Metrics/SSIM', loss["ssim"].detach().data.tolist(), step)           
            summary.add_scalar('Metrics/EDGE', loss["edge"].detach().data.tolist(), step)
            summary.add_scalar('Metrics/Silhouette', loss["silhouette"].detach().data.tolist(), step)
            summary.add_scalar('Metrics/Normal', loss["normal"].detach().data.tolist(), step)
            summary.add_scalar('Metrics/Laplacian', loss["laplacian"].detach().data.tolist(), step)
            summary.add_scalar('Metrics/SUM', sum_loss.detach().data.tolist(), step)
            '''summary.add_scalar('Metrics/gen_loss', gen_loss.detach().data.tolist(), step)
            summary.add_scalar('Metrics/D_train_loss', D_train_loss.detach().data.tolist(), step)'''


            if step%args.delta_test == 0:
                
                ## DRAW WEIGHTS HISTOGRAMS                    
                #draw_weights('mesh', model, summary, epoch)
                ## WRITE IMAGES

                images_predicted = my_render_hard(render_mesh.to(device), trans,global_mat,S)

                predicted_sil = images_predicted[..., 3:].permute(0, 3, 1, 2) > 0.0
                predicted_rgb = images_predicted[..., :3].permute(0, 3, 1, 2) 
                plots_idxs = 0

                summary.add_images('Ground Truth/SIL', seg_soft.detach(), global_step=step, walltime=None)
                summary.add_images('Ground Truth/RGB', img_soft.detach(), global_step=step, walltime=None)
                summary.add_images('Predicted/RGB', predicted_rgb.detach(), global_step=step, walltime=None)
                summary.add_images('Predicted/SIL', predicted_sil.detach(), global_step=step, walltime=None)
                summary.add_images('Grads/MESH', grad_img, global_step=step, walltime=None)

                
 
 
            ## TEST PHASE
            ## VOLTAR IDENTACAO
        if (epoch + 1) % args.save_delta == 0 or (epoch + 1) == args.epochs:       
            print("saving model ...")
            save_model(model.state_dict(), "{}_model{}.pth".format(weights_path, epoch))

        scheduler_mesh.step()
        '''scheduler_tex_D.step()'''

if __name__ == "__main__":
    main()
