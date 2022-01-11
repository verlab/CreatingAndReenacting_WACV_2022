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
from models.p2p_networks import TextureRefinementStage, discriminator, TextureResidualStage, face_discriminator
from utils.mesh_tools import write_obj
from utils.SSIM import SSIM
from utils import arguments
from utils.train_tools import *
from utils.preprocess_noise import Dilation2d, Erosion2d

import random
import config

from pytorch3d.structures import Meshes
from pytorch3d.transforms import Transform3d 

from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
from torch.autograd import Variable

import datetime
import yaml
import pdb

import numpy as np
from tqdm import tqdm

import PIL.Image
from torchvision.transforms import ToTensor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

torch.backends.cudnn.benchmark = True

import io

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

def save_model(state_dict, path):        
    torch.save(state_dict, path)

def read_model(path, device):
    return torch.load(path, map_location = device)

def opening(img_hard, seg_hard, erode, dilate, n_erosions, n_dilations, device, filter_hsv=False):
    img_orig = img_hard.clone()

    ## Getting mask
    background_mask = (seg_hard == 0)
    img_mask = torch.logical_not(background_mask).float()

    ## Criando máscara eroded e dilated.
    seg_eroded = img_mask

    for _ in range(n_erosions):
        seg_eroded = erode(seg_eroded)

    seg_dilated = seg_eroded

    for _ in range(n_dilations):
        seg_dilated = dilate(seg_dilated)

    ## Aplicando erode e dilate em cada canal RGB para fazer borda.
    for channel in range(3):
        img_hard[:, channel:channel+1, ...] = seg_eroded*img_hard[:, channel:channel+1, ...]
        for _ in range(n_dilations): ## Succesion of dilatations in rgb. 
            img_hard[:, channel:channel+1, ...] = dilate(img_hard[:, channel:channel+1, ...])

    ## Imagem RGB recortada no erode com borda dilatada.
    border_mask = (seg_dilated - seg_dilated*seg_eroded) > 0 
    final_img = torch.where(border_mask.repeat(1, 3, 1, 1), img_hard, img_orig*seg_eroded)

    return final_img

def define_bbox(p1, p2, delta=40, size=512):

    min_x = p1-delta
    max_x = p1+delta
    min_y = p2-delta 
    max_y = p2+delta 

    ## Considerando quando bbox esta proximo da borda da imagem.
    ## Checando p1.
    if min_x < 0:
        min_x = 0
        max_x = (p1+delta) + abs(p1-delta)
    if max_x >= size:
        max_x = size - 1
        min_x = (p1-delta) - (p1+delta-size) - 1
    ## Checando p2.
    if min_y < 0:
        min_y = 0
        max_y = (p2+delta) + abs(p2-delta)
    if max_y >= size:
        max_y = size - 1
        min_y = (p2-delta) - (p2+delta-size) - 1

    bbox = (min_x, max_x, min_y, max_y)

    return bbox

def main():
    ###############################################
    ################### CONFIGS ###################
    ###############################################

    ## ARGS
    args = arguments.get_args()

    ## TEXTURE PATH: "/srv/storage/datasets/thiagoluange/dd_dataset/S1P0/tex.jpg"

    ## SUMMARY & CHECKPOINTS
    checkpoint_path = args.dataset_path + '/checkpoints_tex_final/'
    summary_dir = args.dataset_path + '/summaries_tex_final/'

    dataset_person = f"{args.source}{args.person}"

    date = datetime.datetime.now()        
    time_init    = f"{date.day}-{date.month}-{date.year}_{date.hour}:{date.minute}:{date.second}"
    if args.flag is not None:
        time_init = "{}_{}".format(time_init, args.flag)
    summary_path = f"{summary_dir}/{dataset_person}/{args.experiment_name}_texNet-lr_{args.lr_tex}-lr_res_{args.lr_res}-lrdf_{args.lr_d_factor}-lrgb_{args.loss_rgb}-warmup_{args.warm_up}-flip_{args.flip}/batch_{args.batch_size}/epochs_{args.epochs}/{time_init}/"
    weights_path =   f"{checkpoint_path}/{dataset_person}/{args.experiment_name}_texNet-lr_{args.lr_tex}-lr_res_{args.lr_res}-lrdf_{args.lr_d_factor}-lrgb_{args.loss_rgb}-warmup_{args.warm_up}-flip_{args.flip}/batch_{args.batch_size}/epochs_{args.epochs}/{time_init}/"
   
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
    dataloaders = dataloader.get_dataloaders(args, phase = "train")

    ## RECOVERING FIXED PARAMS
    dataset   = dataloaders['train'].dataset
    faces_mesh = torch.from_numpy(np.load(config.SMPL_FACES)).to(device)
    f         = dataset.f
    img_shape = dataset.img_shape

    ## LOAD TEXTURE
    if args.model_texture is None:
        txt_img = np.ones((512, 512, 3))*127
    else:
        txt_img = cv2.resize(cv2.imread(args.model_texture, cv2.IMREAD_UNCHANGED),(512,512))

    ###############################################
    ################# LOAD MODELS #################
    ###############################################
    ## LOAD DILATATION AND EROSION OF IMAGE
    dilate = Dilation2d(in_channels=1, out_channels=1, kernel_size=args.erode_kernel, device=device)
    erode = Erosion2d(in_channels=1, out_channels=1, kernel_size=args.erode_kernel, device=device)

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

    ## LOAD TEXTURE
    model_tex = TextureRefinementStage().to(device)
    model_tex.weight_init(mean=0.0, std=0.02)
    
    model_tex_D = discriminator().to(device)
    model_tex_D.weight_init(mean=0.0, std=0.02)

    model_face = face_discriminator().to(device)
    model_face.weight_init(mean=0.0, std=0.02)
    
    model_tex_res = TextureResidualStage().to(device)
    model_tex_res.weight_init(mean=0.0, std=0.02)

    if(args.pretrained_path_model_tex is not None):
        model_tex.load_state_dict(read_model(args.pretrained_path_model_tex, device))
        print("loaded weights sucessfully")

    ## LOAD RENDER TEXTURE
    my_render_tex = Render_TEX(512, device).to(device)

    ## MODELS OPTIMIZER
    optimizer_tex = torch.optim.AdamW([
                                  #{'params': model.parameters()}
                                   {'params': model_tex.parameters()}
                                 ], lr=args.lr_tex, betas=(0.5, 0.999))

    optimizer_tex_D = torch.optim.AdamW([
                                  #{'params': model.parameters()}
                                   {'params': model_tex_D.parameters()}
                                 ], lr=args.lr_tex/args.lr_d_factor, betas=(0.5, 0.999))

    optimizer_face = torch.optim.AdamW([
                                  #{'params': model.parameters()}
                                   {'params': model_face.parameters()}
                                 ], lr=args.lr_tex/args.lr_d_factor, betas=(0.5, 0.999)) ## Outro teste acontecendo atualmente com: lr=args.lr_tex/args.lr_d_factor

    optimizer_tex_res = torch.optim.AdamW([
                                  #{'params': model.parameters()}
                                   {'params': model_tex_res.parameters()}
                                 ], lr=args.lr_res, betas=(0.5, 0.999))


    #we keep the same learning rate for the first args.epochs/2
    #and linearly decay the rate to zero over the args.epochs/2

    def lambda_rule(epoch):
        lr_l = 1.0 - (epoch/float(args.epochs + 1))
        return lr_l

    scheduler_tex = lr_scheduler.LambdaLR(optimizer_tex, lr_lambda=lambda_rule)
    scheduler_tex_D = lr_scheduler.LambdaLR(optimizer_tex_D, lr_lambda=lambda_rule)
    scheduler_tex_res = lr_scheduler.LambdaLR(optimizer_tex_res, lr_lambda=lambda_rule)
    scheduler_face = lr_scheduler.LambdaLR(optimizer_face, lr_lambda=lambda_rule)

    save_model_epoch = args.save_delta

    step = 0
    best_loss = sys.maxsize

    model_tex.train()
    model_face.train()
    model_tex_D.train()
    model_tex_res.train()

    # create loss
    if args.lsgan:
        BCE_loss = torch.nn.MSELoss().to(device)
    else:
        BCE_loss = torch.nn.BCEWithLogitsLoss().to(device)
    L1_loss = torch.nn.L1Loss(reduction='none').to(device)

    ###############################################
    #################### TRAIN ####################
    ###############################################
    for epoch in range(args.epochs):
        print(f"EPOCH: {epoch}/{args.epochs}") 

        optimizer_tex.zero_grad()
        optimizer_tex_D.zero_grad() ## BATCH

        for idx, (vertices, seg_soft,seg_hard,img_soft,img_hard,trans,global_mat,f_now,face_posi) in enumerate(tqdm(dataloaders['train'])):

            #cv2.imwrite(checkpoint_path + "/image_step_%09d_in"%step + ".jpg", img_hard.permute(0, 2, 3, 1).cpu().detach().numpy()[0,face_posi[1] - 40:face_posi[1] + 40,face_posi[0] - 40:face_posi[0] + 40,-1::-1]*255)
            ###############################################
            ################## MESH OPS ###################
            ###############################################
            transforms = []             
            seg_soft = seg_soft.to(device)
            seg_hard = seg_hard.to(device)
            trans = trans.to(device)
            img_soft = img_soft.to(device)
            img_hard = img_hard.to(device)  
            
            seg_face = torch.zeros((args.batch_size, 1, 80, 80)).to(device)
            img_face = torch.zeros((args.batch_size, 3, 80, 80)).to(device)
            for i in range(len(vertices)):
                min_x, max_x, min_y, max_y = define_bbox(face_posi[1][i], face_posi[0][i]) 
                seg_face[i] = seg_hard[i, :,min_x:max_x,min_y:max_y]
                img_face[i] = img_hard[i, :,min_x:max_x,min_y:max_y]

            img_face = img_face*seg_face
            
            for g_mat, t in zip(global_mat, trans):
                g_mat = g_mat.unsqueeze(0)
                transforms.append(Transform3d(device=device, matrix=torch.transpose(g_mat.view(4,4).to(device),0, 1)).translate(t[0],t[1], t[2]))

            ## CREATE MESH
            vertices = [vert.to(device) for vert in vertices]
            faces = [faces_mesh.to(device) for i in range(len(vertices))] ## O numero de amostras no batch sempre sera batch_size?
            src_mesh = Meshes(verts=vertices, faces=faces).to(device)
            ## DEFORM THE MESH
            with torch.no_grad():
                model.eval()
                subdivide = False
                deformed_mesh = model(src_mesh, subdivide)  
            ## CREATE TEXTURE MAP
            tex_maps = []
            face_normals = []
            _len = deformed_mesh.faces_normals_packed().shape[0]
            it_size = int(_len/len(vertices))
            out = torch.Tensor(len(vertices), 256, 256, 4).to(device)
            for idx in range(0, _len, it_size):
                d_mesh = deformed_mesh.faces_normals_packed()[idx : idx + it_size]
                t_index = int(idx/it_size)
                face_normal = (transforms[t_index].transform_normals(d_mesh)).detach()
                tex_maps.append((my_render_tex(TEX_Mesh(face_normal,device))).detach())
            tex_map = torch.cat(tex_maps, out = out)

            S = torch.ones(f_now.shape[0],3)
            for i in range(f_now.shape[0]):
                S[i,2] = f/f_now[i]
            ###############################################
            ################# TEXTURE OPS #################
            ###############################################          
            #################################################################################################
            ####################################### TRAINING ROUTINES #######################################
            #################################################################################################
            img_hard = img_hard*torch.cat([seg_hard, seg_hard, seg_hard], axis=1)
            
            ###############################################
            ################ DISCRIMINATOR ################
            ###############################################
            if step > args.warm_up:
                model_tex.eval()
                flip = random.random() < args.flip

                with torch.no_grad():
                    txt_img_orig = model_tex(torch.ones_like(torch.cat([tex_map[...,3:],tex_map[...,3:],tex_map[...,3:]], dim=3)))

                txt_img_orig = ( 1 + txt_img_orig )/2

                ## Condicionando residual com segmentacao
                it_size = int(deformed_mesh.verts_packed().shape[0]/len(vertices))
                deformed_meshes = [deformed_mesh.verts_packed()[idx : idx + it_size].detach() for idx in range(0, len(deformed_mesh.verts_packed()), it_size)]
                render_mesh = SMPL_Mesh(deformed_meshes, faces, txt_img_orig, device)
                images_predicted = my_render_hard(render_mesh.to(device), trans, global_mat, S)
                
                predicted_rgb_orig = images_predicted[..., :3] 
                predicted_seg = images_predicted[..., 3:] 
                predicted_seg = (torch.where(predicted_seg < 0.001, predicted_seg, torch.ones_like(predicted_seg))).to(device).detach() ## tem que tirar isso

                ## RESIDUAL
                #txt_img, res_img = model_tex_res(txt_img_orig)
                txt_img, res_img = model_tex_res(torch.cat([txt_img_orig, tex_map[...,3:]], dim=3))
                txt_img = ( 1 + txt_img )/2

                #txt_img, res_img = model_tex_res(txt_img_orig*torch.cat([tex_map[...,3:],tex_map[...,3:],tex_map[...,3:]], dim=3))
                #txt_img, res_img = model_tex_res(torch.cat([tex_map[...,3:],tex_map[...,3:],tex_map[...,3:]], dim=3))
                it_size = int(deformed_mesh.verts_packed().shape[0]/len(vertices))
                deformed_meshes = [deformed_mesh.verts_packed()[idx : idx + it_size].detach() for idx in range(0, len(deformed_mesh.verts_packed()), it_size)]
                render_mesh = SMPL_Mesh(deformed_meshes, faces, txt_img, device)
                
                
                images_predicted = my_render_hard(render_mesh.to(device), trans, global_mat, S)
                
                
                predicted_rgb = images_predicted[..., :3]  
                predicted_seg = images_predicted[..., 3:] 
                predicted_seg = (torch.where(predicted_seg < 0.001, predicted_seg, torch.ones_like(predicted_seg))).to(device).detach().permute(0, 3, 1, 2) ## tem que tirar isso

                mask_l1 = (predicted_seg*seg_hard).to(device)
                predicted_rgb = predicted_rgb.permute(0, 3, 1, 2)*torch.cat([predicted_seg, predicted_seg, predicted_seg], dim=1).to(device)

                ## INIT VARS
                
                predicted_seg_face = torch.zeros((args.batch_size, 1, 80, 80)).to(device)
                predicted_seg_face.retain_grad()
                predicted_face = torch.zeros((args.batch_size, 3, 80, 80)).to(device)
                predicted_face.retain_grad()
                not_face = predicted_seg.detach().clone()
                for i in range(len(vertices)):
                    min_x, max_x, min_y, max_y = define_bbox(face_posi[1][i], face_posi[0][i]) 
                    predicted_seg_face[i] = predicted_seg[i, :,min_x:max_x,min_y:max_y]
                    predicted_face[i] = predicted_rgb[i, :,min_x:max_x,min_y:max_y,]
                    not_face[i, :,min_x:max_x,min_y:max_y] = 0

                predicted_face = predicted_face*predicted_seg_face                
                        
                pred_fake = torch.cat([predicted_rgb.detach()], axis=1)
                face_fake = torch.cat([predicted_face.detach()], axis=1)
                #pred_fake = predicted_rgb.detach()*torch.cat([predicted_seg, predicted_seg, predicted_seg], axis=1)
                real = torch.cat([img_hard], axis=1)
                face_real = torch.cat([img_face], axis=1)
                #real = img_soft.detach()*torch.cat([seg_soft, seg_soft, seg_soft], axis=1)

                # train with real
                optimizer_tex_D.zero_grad()
                optimizer_face.zero_grad()

                if flip:
                    D_result = model_tex_D(pred_fake).squeeze()
                    D_result_face = model_face(face_fake).squeeze()
                else:
                    D_result = model_tex_D(real).squeeze()
                    D_result_face = model_face(face_real).squeeze()

                valid = torch.Tensor(np.random.uniform(low=1, high=1, size=D_result.size())).to(device)
                valid_face = torch.Tensor(np.random.uniform(low=1, high=1, size=D_result_face.size())).to(device)

                D_real_loss = BCE_loss(D_result, valid)
                D_real = D_result.mean().item()

                D_real_loss1 = BCE_loss(D_result_face, valid_face)
                D_real1 = D_result_face.mean().item()
                #D_real_loss.backward()

                ## train with fake
                if flip:
                    D_result = model_tex_D(real).squeeze()
                    D_result_face = model_face(face_real).squeeze()
                else:
                    D_result = model_tex_D(pred_fake).squeeze()
                    D_result_face = model_face(face_fake).squeeze()

                fake = torch.Tensor(np.random.uniform(low=0.0, high=0, size=D_result.size())).to(device)
                fake_face = torch.Tensor(np.random.uniform(low=0.0, high=0, size=D_result_face.size())).to(device)

                D_fake_loss = BCE_loss(D_result, fake)
                D_fake_loss1 = BCE_loss(D_result_face, fake_face)
                D_fake1 = D_result.mean().item()
                #D_fake_loss.backward()

                # Discr loss
                D_train_loss = (D_fake_loss + D_real_loss)
                D_train_loss.backward()

                D_train_loss1 = (D_fake_loss1 + D_real_loss1)
                D_train_loss1.backward()

                if step > args.warm_up:
                    optimizer_tex_D.step()
                    optimizer_face.step()
            ###############################################
            ################## GENERATOR ##################
            ###############################################
            if step > args.warm_up: ## warm start
                optimizer_tex_D.zero_grad()
                optimizer_face.zero_grad()
                optimizer_tex_res.zero_grad()
                
                gen_input = torch.cat([predicted_rgb], axis=1)
                D_result = model_tex_D(gen_input).squeeze()

                gen_input1 = torch.cat([predicted_face], axis=1)
                D_result1 = model_face(gen_input1).squeeze()
                #D_result = model_tex_D(predicted_rgb*torch.cat([predicted_seg, predicted_seg, predicted_seg], axis=1)).squeeze()
                gen_loss = BCE_loss(D_result, valid)

                gen_loss_face = BCE_loss(D_result1, valid_face)

                D_fake2 = D_result.mean().item()
                
                loss_rgb = (torch.sum(L1_loss(predicted_rgb,img_hard)*torch.cat([mask_l1, mask_l1, mask_l1], dim=1))/torch.sum(1*(mask_l1.detach().flatten() == 1)))
                
                mask_mismatch = torch.where((predicted_seg-seg_hard) > 0, torch.ones_like(predicted_seg), torch.zeros_like(predicted_seg))*not_face.to(device)
                vanish_loss = (torch.sum(L1_loss(predicted_rgb, predicted_rgb_orig.permute(0, 3, 1, 2))*torch.cat([mask_mismatch, mask_mismatch, mask_mismatch], dim=1))/torch.sum(1*(mask_mismatch.detach().flatten() == 1)))

                ##
                G_train_loss = loss_rgb*args.loss_rgb + gen_loss + gen_loss_face + 100*vanish_loss

                G_train_loss.backward()

                if step%args.delta_test == 0:
                    res_grad_img = get_grad_img(model_tex_res)

                optimizer_tex_res.step()
            else:
                optimizer_tex.zero_grad()
                txt_img_orig = model_tex(torch.ones_like(torch.cat([tex_map[...,3:],tex_map[...,3:],tex_map[...,3:]], dim=3)))
                txt_img_orig = ( 1 + txt_img_orig )/2
                it_size = int(deformed_mesh.verts_packed().shape[0]/len(vertices))
                deformed_meshes = [deformed_mesh.verts_packed()[idx : idx + it_size].detach() for idx in range(0, len(deformed_mesh.verts_packed()), it_size)]
                render_mesh = SMPL_Mesh(deformed_meshes, faces, txt_img_orig, device)
                images_predicted = my_render_hard(render_mesh.to(device), trans, global_mat, S)
                
                predicted_rgb = images_predicted[..., :3]
                predicted_seg = images_predicted[..., 3:] 
                predicted_seg = (torch.where(predicted_seg < 0.001, predicted_seg, torch.ones_like(predicted_seg))).to(device).detach().permute(0, 3, 1, 2) ## tem que tirar isso
                
                mask_l1 = (predicted_seg*seg_hard).to(device)

                predicted_rgb = predicted_rgb.permute(0, 3, 1, 2)*torch.cat([predicted_seg, predicted_seg, predicted_seg], dim=1).to(device)

                loss_rgb = (torch.sum(L1_loss(predicted_rgb,img_hard)*torch.cat([mask_l1, mask_l1, mask_l1], dim=1))/torch.sum(1*(mask_l1.detach().flatten() == 1)))
                G_train_loss = loss_rgb*args.loss_rgb

                G_train_loss.backward()
                if step%args.delta_test == 0:
                    tex_grad_img = get_grad_img(model_tex)

                optimizer_tex.step()

            ###############################################
            ################# TENSORBOARD #################
            ###############################################
            if step > args.warm_up and args.gan:
                summary.add_scalar('Metrics/G_train_loss', G_train_loss.detach().data.tolist(), step)
                summary.add_scalar('Metrics/D_train_loss', D_train_loss.detach().data.tolist(), step)
                summary.add_scalar('Metrics/D_face_loss', D_train_loss1.detach().data.tolist(), step)
                summary.add_scalar('Metrics/gen_loss', gen_loss.detach().data.tolist(), step)
                summary.add_scalar('Metrics/Prob_real', D_real, step)
                summary.add_scalar('Metrics/Prob_fake1', D_fake1, step)
                summary.add_scalar('Metrics/Prob_fake2', D_fake2, step)
                summary.add_scalar('Metrics/Vanish_loss', vanish_loss.detach().data.tolist(), step)
            summary.add_scalar('Metrics/RGB', loss_rgb.detach().data.tolist(), step)

            if step%args.delta_test == 0:
                predicted_sil = predicted_seg

                plots_idxs = 0
                ## WRITE IMAGES
                summary.add_images('Ground Truth/SIL', seg_hard.detach(), global_step=step, walltime=None)
                summary.add_images('Ground Truth/RGB', img_hard.detach(), global_step=step, walltime=None)
                summary.add_images('Predicted/RGB', predicted_rgb.detach(), global_step=step, walltime=None)
                summary.add_images('Predicted/SIL', predicted_sil.detach(), global_step=step, walltime=None)
                summary.add_images('Predicted/TEX_ORIG', txt_img_orig.permute(0, 3, 1, 2).detach(), global_step=step, walltime=None)
                summary.add_images('Grads/TEX', tex_grad_img, global_step=step, walltime=None)
                
                if step > args.warm_up and args.gan:
                    predicted_rgb_orig = predicted_rgb_orig.permute(0, 3, 1, 2)*torch.cat([predicted_seg, predicted_seg, predicted_seg], dim=1)
                    summary.add_images('Face/FACE_REAL', img_face.detach(), global_step=step, walltime=None)
                    summary.add_images('Face/FACE_PRED', predicted_face.detach(), global_step=step, walltime=None)
                    summary.add_images('Predicted/RGB0', predicted_rgb_orig.detach(), global_step=step, walltime=None)
                    summary.add_images('Predicted/TEX_RES', res_img.permute(0, 3, 1, 2).detach(), global_step=step, walltime=None)
                    summary.add_images('Grads/RES', res_grad_img, global_step=step, walltime=None)
                    
            step = step + 1

        if (epoch + 1) % save_model_epoch == 0 or epoch == args.epochs:            
            print("saving model ...")
            save_model(model.state_dict(), "{}_model.pth".format(weights_path))
            save_model(model_tex.state_dict(), "{}_model_tex.pth".format(weights_path))
            save_model(model_tex_res.state_dict(), "{}_model_tex_res_{}.pth".format(weights_path, epoch))

        scheduler_tex.step()
        scheduler_tex_D.step()
        scheduler_tex_res.step()
        scheduler_face.step()

    ###############################################
    #################### TEST #####################
    ###############################################
    if args.test:
        test(dataloader, f, img_shape, device, model, model_tex, summary)

if __name__ == "__main__":
    main()