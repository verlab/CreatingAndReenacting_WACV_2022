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
import pdb

import numpy as np
from tqdm import tqdm

# Losses to smooth / regularize the mesh shape
def update_mesh_shape_prior_losses(mesh,loss):
    # and (b) the edge length of the predicted mesh
    #loss["edge"] = mesh_edge_loss(mesh)

    edges_packed = mesh.edges_packed() 
    verts_packed = mesh.verts_packed()

    verts_edges = verts_packed[edges_packed]
    v0, v1 = verts_edges.unbind(1)

    edge_size = ((v0 - v1).norm(dim=1, p=2)) ** 2.0 - 0.0025

    m = torch.nn.ReLU()
 
    loss["edge"] = (m(edge_size)).sum()

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



#python train.py -d /media/thiagoluange/SAMSUNG/ -w 1 -rss 50 -rsh 75 

def main():

    ## ARGS
    args = arguments.get_args()

    ## TEXTURE PATH: "/srv/storage/datasets/thiagoluange/dd_dataset/S1P0/tex.jpg"

    ## SUMMARY & CHECKPOINTS
    checkpoint_path = args.dataset_path + '/checkpoints_mesh_gan/'
    summary_dir = args.dataset_path + '/summaries_mesh_gan/'

    dataset_person = f"{args.source}{args.person}"

    date = datetime.datetime.now()        
    time_init    = f"{date.day}-{date.month}-{date.year}_{date.hour}:{date.minute}:{date.second}"
    if args.flag is not None:
        time_init = "{}_{}".format(time_init, args.flag)
    summary_path = f"{summary_dir}/{dataset_person}/BIGGERLR_BATCH_WARMSTARTmeshNet_lr_{args.lr}-lrgb_{args.loss_rgb}-lss_{args.loss_ssim}-ls_{args.loss_sil}-ll_{args.loss_lap}-ln_{args.loss_nor}/batch_{args.batch_size}/epochs_{args.epochs}/{time_init}/"
    weights_path =   f"{checkpoint_path}/{dataset_person}/BIGGERLR_BATCH_WARMSTARTmeshNet_lr_{args.lr}-lrgb_{args.loss_rgb}-lss_{args.loss_ssim}-ls_{args.loss_sil}-ll_{args.loss_lap}-ln_{args.loss_nor}/batch_{args.batch_size}/epochs_{args.epochs}/{time_init}/"
   
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
    faces_mesh = dataset.faces
    f         = dataset.f
    '''
        NOTE: Atualmente img_shape recebe o tamanho da imagem original. 
        Talvez tenha que mudar para tamanho da imagem cropada.
        Seria pegar a menor dimensao para montar imagem quadrada? Thiago: Deve receber o tamanho original mesmo, ele que defini a camera
    '''
    img_shape = dataset.img_shape

    ## LOAD TEXTURE
    if args.model_texture is None:
        txt_img = np.ones((512, 512, 3))*128
    else:
        txt_img = cv2.resize(cv2.imread(args.model_texture, cv2.IMREAD_UNCHANGED),(512,512))
    '''
        NOTE: Nao sei o quanto o tamanho da textura vai influenciar na qualidade e no peso da rede
    '''

    ### LOADING MODELS
    ## LOAD MESH MODEL
    with open("models/model_cfg.yaml", 'r') as cfg_file:
        model_cfgs = yaml.safe_load(cfg_file)

    model_cfgs["device"] = device
    model = MeshRefinementHead(model_cfgs).to(device)
    if(args.pretrained_path_model is not None):
        model.load_state_dict(read_model(args.pretrained_path_model, device))
        print("loaded weights sucessfully")

    ## LOAD RENDER MODEL
    my_render_soft = Render_SMPL(f, img_shape, args.render_size_soft, device).to(device)     
    my_render_hard = Render_SMPL(f, img_shape, args.render_size_hard, device, "hard").to(device) 


    ## LOAD TEXTURE
    model_tex = TextureRefinementStage().to(device)
    model_tex.weight_init(mean=0.0, std=0.02)
    
    model_tex_D = discriminator().to(device)
    model_tex_D.weight_init(mean=0.0, std=0.02)

    if(args.pretrained_path_model_tex is not None):
        model_tex.load_state_dict(read_model(args.pretrained_path_model_tex, device))
        print("loaded weights sucessfully")

    ## LOAD RENDER TEXTURE
    my_render_tex = Render_TEX(256,device).to(device)
  
    ## MODELS OPTIMIZER
    optimizer_mesh = torch.optim.Adam([
                                    {'params': model.parameters()}
                                 #   {'params': model_tex.parameters()}
                                 ], lr=args.lr, betas=(0.5, 0.999))
    optimizer_tex = torch.optim.Adam([
                                  #{'params': model.parameters()}
                                   {'params': model_tex.parameters()}
                                 ], lr=args.lr_tex, betas=(0.5, 0.999))

    optimizer_tex_D = torch.optim.SGD([
                                  #{'params': model.parameters()}
                                   {'params': model_tex_D.parameters()}
                                 ], lr=args.lr_tex/1000)


    #we keep the same learning rate for the first args.epochs/2
    #and linearly decay the rate to zero over the args.epochs/2
    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch - args.epochs/2) / float(args.epochs/2 + 1)
        return lr_l

    scheduler_mesh = lr_scheduler.LambdaLR(optimizer_mesh, lr_lambda=lambda_rule)
    scheduler_tex = lr_scheduler.LambdaLR(optimizer_tex, lr_lambda=lambda_rule)
    scheduler_tex_D = lr_scheduler.LambdaLR(optimizer_tex_D, lr_lambda=lambda_rule)


    ## SETTING LOSSES
    losses = {"silhouette": {"weight": args.loss_sil},
        "ssim": {"weight": args.loss_ssim},
        "edge": {"weight": args.loss_edge},
        "normal": {"weight": args.loss_nor},
        "laplacian": {"weight": args.loss_lap},
    }

    save_model_step = int((len(dataloaders['train'].dataset)/args.batch_size)*10000)

    step = 0
    best_loss = sys.maxsize
    #dataset_test = list(dataloader_test["test"])

    

    ## MODELS TRAIN
    model.train()
    model_tex.train()
    model_tex_D.train()

    # create loss
    BCE_loss = torch.nn.BCELoss().to(device)
    L1_loss = torch.nn.L1Loss(reduction='none').to(device)
    model_ssim = SSIM().to(device)

    for epoch in range(args.epochs):
        print(f"EPOCH: {epoch}/{args.epochs}") 

        optimizer_tex.zero_grad()
        optimizer_tex_D.zero_grad() ## BATCH

        for idx, (vertices, seg_soft,seg_hard,img_soft,img_hard,trans,global_mat) in enumerate(tqdm(dataloaders['train'])):

            ############            Initialize optimizer mesh             
            optimizer_mesh.zero_grad()
            seg_soft = seg_soft.to(device)
            seg_hard = seg_hard.to(device)
            trans = trans.to(device)
            img_soft = img_soft.to(device)
            img_hard = img_hard.to(device)  
   
            my_transform = Transform3d(device=device, matrix=torch.transpose(global_mat.view(4,4).to(device),0, 1)).translate(trans[0,0],trans[0,1], trans[0,2])
            #verts_camera = my_transform.transform_points(vertices.to(device))

            ## CREATE MESH
            vertices = [vert.to(device) for vert in vertices]
            faces = [faces_mesh.to(device) for i in range(len(vertices))] ## O numero de amostras no batch sempre sera batch_size?
            src_mesh = Meshes(verts=vertices, faces=faces).to(device)
         
          
            # Deform the mesh
            with torch.no_grad():
                model.eval()
                subdivide = False
                deformed_mesh = model(src_mesh, subdivide)  
       
            ## CREATE texture map
            face_normal = (my_transform.transform_normals(deformed_mesh.faces_normals_packed())).detach()
            
            tex_map = (my_render_tex(TEX_Mesh(face_normal,device))).detach()
            #cv2.imwrite(checkpoint_path + "/image_pred_step_%09d"%step + ".jpg", tex_map.cpu().detach().numpy()[0,:,:,3:]*255)

            #pdb.set_trace()
         
            '''txt_img = model_tex(torch.cat([tex_map[...,3:],tex_map[...,3:],tex_map[...,3:]], dim=3))[0]


            render_mesh = SMPL_Mesh([deformed_mesh.verts_packed()], faces, txt_img, device)
            

            #Losses to smooth /regularize the mesh shape
            loss = {k: torch.tensor(0.0, device=device) for k in losses}
            
            update_mesh_shape_prior_losses(deformed_mesh,loss)

            images_predicted = my_render_soft(render_mesh.to(device), trans,global_mat)

            num_views_per_iteration = img_soft.shape[0]

            predicted_silhouette = images_predicted[..., 3:].to(device)'''

            #loss_silhouette = ((predicted_silhouette - seg.permute(0, 2, 3, 1)) ** 2).mean()

            '''loss_silhouette =  torch.tensor(1.0, device=device) - torch.norm(predicted_silhouette*seg_soft.permute(0, 2, 3, 1),1)/torch.norm(predicted_silhouette + seg_soft.permute(0, 2, 3, 1) - predicted_silhouette*seg_soft.permute(0, 2, 3, 1),1)

            loss_ssim = 1.0 - model_ssim(seg_soft,predicted_silhouette.permute(0, 3, 1, 2))        

            loss["ssim"] +=  loss_ssim / num_views_per_iteration

            loss["silhouette"] += loss_silhouette / num_views_per_iteration            
           

            # Weighted sum of the losses
            sum_loss = torch.tensor(0.0, device=device)

            for k, l in loss.items():
                sum_loss += l * losses[k]["weight"]
            # Print the losses
            sum_loss = sum_loss/3
          
            # Optimization step

            sum_loss.backward()
            optimizer_mesh.step()'''

            ############            End optimizer mesh    


            ###########           Initialize optimizer tex   ###############

            #forward D

            mask = torch.cat([seg_hard, seg_hard, seg_hard], dim=1)   
            img_hard = img_hard*mask + torch.ones_like(mask) - mask
            txt_img = model_tex(torch.cat([tex_map[...,3:],tex_map[...,3:],tex_map[...,3:]], dim=3))[0]
            render_mesh = SMPL_Mesh([deformed_mesh.verts_packed().detach()], faces, txt_img, device)
            images_predicted = my_render_hard(render_mesh.to(device), trans,global_mat)
            predicted_rgb = images_predicted[..., :3].to(device)  
            predicted_seg = images_predicted[..., 3:] 
            predicted_seg = (torch.where(predicted_seg < 0.001, predicted_seg, torch.ones_like(predicted_seg))).to(device).detach()
           
            ############# discriminator ################
            if step > 1000:
                flip = random.random() > 0.7

                if flip: ## Passando fake como real
                    pred_fake = torch.cat([(predicted_rgb.detach().permute(0, 3, 1, 2)),predicted_seg.permute(0, 3, 1, 2)],dim=1)            
                    D_result = model_tex_D(pred_fake).squeeze()
                else: ## Passando real normal
                    D_result =  model_tex_D(torch.cat([img_hard,seg_hard],dim=1)).squeeze()
                valid = torch.Tensor(np.random.uniform(low=0.7, high=1.2, size=D_result.size())).to(device)
                D_real_loss = BCE_loss(D_result, valid)

                # Fake; stop backprop to the generator by detaching fake_B

                if flip: ## Passando real como fake
                    D_result =  model_tex_D(torch.cat([img_hard,seg_hard],dim=1)).squeeze()
                else: ## Passando fake normal
                    pred_fake = torch.cat([(predicted_rgb.detach().permute(0, 3, 1, 2)),predicted_seg.permute(0, 3, 1, 2)],dim=1)            
                    D_result = model_tex_D(pred_fake).squeeze()

                fake = torch.Tensor(np.random.uniform(low=0.0, high=0.3, size=D_result.size())).to(device)
                D_fake_loss = BCE_loss(D_result, fake)

                D_train_loss = (D_real_loss + D_fake_loss) * 0.5
                D_train_loss.backward()

                if (step+1)%16 == 0:
                    optimizer_tex_D.step()
                    optimizer_tex_D.zero_grad() ## BATCH


            ############# end discriminator ################

            ############# Generator ################

            if step > 1000: ## warm start
                pred_fake = torch.cat([(predicted_rgb.permute(0, 3, 1, 2)),predicted_seg.permute(0, 3, 1, 2)],dim=1)
                D_result = model_tex_D(pred_fake).squeeze()

                loss_rgb = (L1_loss(predicted_rgb,img_hard.permute(0, 2, 3, 1))* torch.cat([seg_hard.permute(0, 2, 3, 1),seg_hard.permute(0, 2, 3, 1),seg_hard.permute(0, 2, 3, 1)], dim=3)).mean()
                gen_loss = BCE_loss(D_result, valid)
                G_train_loss = gen_loss + args.loss_rgb*loss_rgb
            else:
                loss_rgb = (L1_loss(predicted_rgb,img_hard.permute(0, 2, 3, 1))* torch.cat([seg_hard.permute(0, 2, 3, 1),seg_hard.permute(0, 2, 3, 1),seg_hard.permute(0, 2, 3, 1)], dim=3)).mean()
                G_train_loss = args.loss_rgb*loss_rgb
  
            G_train_loss.backward()
            if (step+1)%16 == 0:
                optimizer_tex.step()
                optimizer_tex.zero_grad() ## BATCH

            ###########           End optimizer tex




            #cv2.imwrite(checkpoint_path + "/image_pred_step_%09d"%step + ".jpg", predicted_rgb.cpu().detach().numpy()[0,:,:,-1::-1]*255)
            #cv2.imwrite(checkpoint_path + "/image_real_step_%09d"%step + ".jpg", img.permute(0, 2, 3, 1).cpu().detach().numpy()[0,:,:,-1::-1]*255)

            #cv2.imwrite(checkpoint_path + "/seg_step_%09d"%step + ".jpg", predicted_silhouette.cpu().detach().numpy()[0,:,:,:]*255)

            ## UPDATING TRAINING LOSSES

            #loop.set_description("total_loss = %.6f" % sum_loss)
            #tqdm.write("total_loss = %.6f" % float(0))
            '''if(sum_loss < best_loss and args.save_best_loss):
                print("saving best loss model")
                save_model(model.state_dict(), "{}_model.pth".format(weights_path))
                save_model(model_tex.state_dict(), "{}_model_tex.pth".format(weights_path))
                best_loss = sum_loss
            '''
            step = step + 1
            #summary.add_scalar('Metrics/SSIM', loss["ssim"].detach().data.tolist(), step)
            if step > 1001:
                summary.add_scalar('Metrics/G_train_loss', G_train_loss.detach().data.tolist(), step)
                summary.add_scalar('Metrics/D_train_loss', D_train_loss.detach().data.tolist(), step)
                summary.add_scalar('Metrics/gen_loss', gen_loss.detach().data.tolist(), step)
            #summary.add_scalar('Metrics/EDGE', loss["edge"].detach().data.tolist(), step)
            #summary.add_scalar('Metrics/Silhouette', loss["silhouette"].detach().data.tolist(), step)
            summary.add_scalar('Metrics/RGB', loss_rgb.detach().data.tolist(), step)
            #summary.add_scalar('Metrics/Normal', loss["normal"].detach().data.tolist(), step)
            #summary.add_scalar('Metrics/Laplacian', loss["laplacian"].detach().data.tolist(), step)
            #summary.add_scalar('Metrics/SUM', sum_loss.detach().data.tolist(), step)

            if step%args.delta_test == 0:
                predicted_sil = predicted_seg.permute(0, 3, 1, 2)
                predicted_rgb = predicted_rgb.permute(0, 3, 1, 2)
                plots_idxs = 0
                ## DRAW WEIGHTS HISTOGRAMS                    
                #draw_weights('mesh', model, summary, epoch)
                ## WRITE IMAGES
                summary.add_images('Ground Truth/SIL', seg_soft.detach(), global_step=step, walltime=None)
                summary.add_images('Ground Truth/RGB', img_hard.detach(), global_step=step, walltime=None)
                summary.add_images('Predicted/RGB', predicted_rgb.detach(), global_step=step, walltime=None)
                summary.add_images('Predicted/SIL', predicted_sil.detach(), global_step=step, walltime=None)
                
 
 
            ## TEST PHASE
            ## VOLTAR IDENTACAO
            if (step + 1) % save_model_step == 0:            
                print("saving model ...")
                save_model(model.state_dict(), "{}_model.pth".format(weights_path))
                save_model(model_tex.state_dict(), "{}_model_tex.pth".format(weights_path))

        scheduler_mesh.step()
        scheduler_tex.step()
        scheduler_tex_D.step()

    ## TEST TRAINED MODEL ON CONE AND BOX VIDEOS
    if args.test:
        print("Testing trained model on cone and box videos...")
        movements = ['bruno', 'box', 'cone']
        dataloaders = dataloader.get_dataloaders(args, phase = "test", movements=movements)

        video = []
    
        model.eval()
        model_tex.eval()
        my_render_hard = Render_SMPL(f, img_shape, 1000, device, "hard").to(device)

        for idx, (vertices, trans,global_mat) in enumerate(tqdm(dataloaders['test'])):
            
            vertices = [vert.to(device) for vert in vertices]
            faces = [faces_mesh.to(device) for i in range(len(vertices))] 
            trans = trans.to(device)
         
            with torch.no_grad():
                ## CREATE MESH
                src_mesh = Meshes(verts=vertices, faces=faces).to(device)                
                subdivide = False
                deformed_mesh = model(src_mesh, subdivide) 
                my_transform = Transform3d(device=device, matrix=torch.transpose(global_mat.view(4,4).to(device),0, 1)).translate(trans[0,0],trans[0,1], trans[0,2]) 
       
                ## CREATE texture map
                face_normal = (my_transform.transform_normals(deformed_mesh.faces_normals_packed())).detach()
                tex_map = (my_render_tex(TEX_Mesh(face_normal,device))).detach()
                txt_img = model_tex(torch.cat([tex_map[...,3:],tex_map[...,3:],tex_map[...,3:]], dim=3))[0] 
              
                ## CREATE MESH with texture  
                render_mesh = SMPL_Mesh([deformed_mesh.verts_packed()], faces, txt_img, device)
            
                ## RENDER
                images_predicted = my_render_hard(render_mesh.to(device), trans,global_mat)
                predicted_rgb = images_predicted[..., :3].cpu().detach()

                video.append(predicted_rgb.permute(0, 3, 1, 2)*255)

        video = torch.cat(video).unsqueeze(0)
        video = video.type(torch.uint8)
        video = video.detach()
        summary.add_video(tag='Test Video', vid_tensor=video, global_step=0, fps=30)
        summary.flush()

if __name__ == "__main__":
    main()
