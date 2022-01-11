import os
import sys
from utils import dataloader
import torch
import torchvision
import cv2
from models.render import Render_SMPL,Render_TEX
from models.mesh import SMPL_Mesh,TEX_Mesh
from models.smpl import SMPL,load_smpl
from models.meshNet import MeshRefinementStage, MeshRefinementHead
from models.p2p_networks import TextureRefinementStage, TextureResidualStage
from utils.mesh_tools import write_obj
from utils.SSIM import SSIM
from utils import arguments
from pytorch3d.transforms import Transform3d

from pytorch3d.utils import ico_sphere
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)
from pytorch3d.structures import Meshes

import config

import datetime
import yaml
import pdb

import numpy as np
from tqdm import tqdm
from pytorch3d.io import save_obj

def write_obj(out_name,out_vert,out_faces,verts_uvs,faces_uvs,texture_map,i):

    cv2.imwrite(out_name + ".jpg", texture_map.cpu().detach().numpy()[i,:,:,-1::-1]*255)

    with open(out_name + ".mtl", 'w') as my_file:
            my_file.write("newmtl Material.001" + "\n" + "Ns 96.078431" + "\n" + "Ka 1.000000 1.000000 1.000000" + "\n" + "Kd 0.640000 0.640000 0.640000" + "\n" + "Ks 0.000000 0.000000 0.000000" + "\n" + "Ke 0.0 0.0 0.0" + "\n" + "Ni 1.000000" + "\n" + "d 1.000000" + "\n" + "illum 1" + "\n" + "map_Kd " +  out_name + ".jpg")

    with open(out_name + ".obj", 'w') as my_file:
        my_file.write("# OBJ file\n")
        my_file.write("mtllib " + out_name + ".mtl\n") 
        my_file.write("o " + out_name.split("/")[-1] + "\n")
        for v in range(out_vert.shape[0]):
            my_file.write("v " + str(float(out_vert[v][0])) + " " + str(float(out_vert[v][1])) + " " + str(float(out_vert[v][2])) + "\n" )
        for vt in range(verts_uvs.shape[0]):
            my_file.write("vt " + str(verts_uvs[vt][0]) + " " + str(verts_uvs[vt][1]) + "\n" )
       
        my_file.write("usemtl Material.001" + "\n" + "s off" + "\n")

        for f in range(out_faces.shape[0]):
            my_file.write("f " + str(int(out_faces[f][0]) + 1) + "/" +  str(faces_uvs[f][0] + 1) + " "  + str(int(out_faces[f][1]) + 1) + "/" +  str(faces_uvs[f][1] + 1) + " "  + str(int(out_faces[f][2]) + 1) + "/" +  str(faces_uvs[f][2] + 1) + "\n" )






def read_model(path, device):
    return torch.load(path, map_location = device)

def main():

    args = arguments.get_args()

    ## GETTING DEVICE
    if(torch.cuda.is_available()):
        device = torch.device("cuda:{}".format(args.device))
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    print(f"RUNNING ON {device}")
    
    #dataset   = dataloaders['test'].dataset
    faces_mesh = torch.from_numpy(np.load(config.SMPL_FACES)).to(device)

    ### LOADING MODELS
    ## LOAD MESH MODEL
    with open("models/model_cfg.yaml", 'r') as cfg_file:
        model_cfgs = yaml.safe_load(cfg_file)

    model_cfgs["device"] = device
    model_cfgs["batch_size"] = args.batch_size
    model = MeshRefinementHead(model_cfgs).to(device)
    if(args.pretrained_path_model is not None):
        model.load_state_dict(read_model(args.pretrained_path_model, device))
        print("[MESH MODEL] loaded weights sucessfully")

    ## LOAD RENDER MODEL
    dataloaders, dataset = dataloader.get_dataloaders(args, phase = "test", movements=['box'], test=True)
    img_shape = dataset.img_shape
    f         = dataset.f

    my_render_soft = Render_SMPL(f, img_shape, args.render_size_soft, device).to(device)     
    my_render_hard = Render_SMPL(f, img_shape, 512, device, "hard").to(device) 

    ## LOAD TEXTURE
    model_tex = TextureRefinementStage().to(device)
    if(args.pretrained_path_model_tex is not None): 
        model_tex.load_state_dict(read_model(args.pretrained_path_model_tex, device))
        print("[TEX MODEL] loaded weights sucessfully")

    model_tex_res = TextureResidualStage().to(device)
    if(args.pretrained_path_model_tex_res is not None):
        model_tex_res.load_state_dict(read_model(args.pretrained_path_model_tex_res, device))
        print("[TEX_RES MODEL] loaded weights sucessfully")

    ## LOAD RENDER TEXTURE
    my_render_tex = Render_TEX(512,device).to(device)

    output_path = args.output_path

    model.eval()
    model_tex.eval()
    model_tex_res.eval()

    #movements = [args.movement]
    #movements = ["box", "cone", "fusion", "hand", "jump", "rotate", "shake_hands", "simple_walk"]
    movements = [args.movement]
    mask_list = []
    for movement in tqdm(movements, desc="Testando videos..."):
        dataloaders, dataset = dataloader.get_dataloaders(args, phase = "test", movements=[movement], test=True)

        output_path_rgb_pred = os.path.join(output_path, args.source, movement, args.test_person, "rgb")
        output_path_mask_pred = os.path.join(output_path, args.source, movement, args.test_person, "mask")
        #output_path_sil_pred = os.path.join(output_path, args.source, args.person, args.movement, "sil")

        os.makedirs(output_path_rgb_pred, exist_ok=True)
        os.makedirs(output_path_mask_pred, exist_ok=True)
        if args.save_mesh_texture:
            output_path_texture_pred = os.path.join(output_path, args.source, movement, args.test_person, "texture")
            output_path_mesh_pred = os.path.join(output_path, args.source, movement, args.test_person, "mesh")
            os.makedirs(output_path_texture_pred, exist_ok=True)
            os.makedirs(output_path_mesh_pred, exist_ok=True)
        #os.makedirs(output_path_sil_pred, exist_ok=True)
        
        step = 0

        for idx, (vertices, trans,global_mat,f_now) in enumerate(dataloaders['test']):

            vertices = [vert.to(device) for vert in vertices]
            faces = [faces_mesh.to(device) for i in range(len(vertices))] 
            
            with torch.no_grad():
                ## CREATE MESH
                src_mesh = Meshes(verts=vertices, faces=faces).to(device)                
                subdivide = False
                with torch.no_grad():
                    deformed_mesh = model(src_mesh, subdivide) 
                transforms = [] 
                for g_mat, t in zip(global_mat, trans):
                    g_mat = g_mat.unsqueeze(0)
                    transforms.append(Transform3d(device=device, matrix=torch.transpose(g_mat.view(4,4).to(device),0, 1)).translate(t[0],t[1], t[2]))
        
                ## CREATE texture map
                tex_maps = []
                face_normals = []
                _len = deformed_mesh.faces_normals_packed().shape[0]
                it_size = int(_len/len(vertices))
                out = torch.Tensor(len(vertices), 512, 512, 4).to(device)
                for idx in range(0, _len, it_size):
                    d_mesh = deformed_mesh.faces_normals_packed()[idx : idx + it_size]
                    t_index = int(idx/it_size)
                    face_normal = (transforms[t_index].transform_normals(d_mesh)).detach()
                    tex_maps.append((my_render_tex(TEX_Mesh(face_normal,device))).detach())

                tex_map = torch.cat(tex_maps, out = out)

                S = torch.ones(f_now.shape[0],3)
                for i in range(f_now.shape[0]):
                    S[i,2] = f/f_now[i]

                
                ## Create initial texture
                with torch.no_grad():
                    txt_img_orig = model_tex(torch.ones_like(torch.cat([tex_map[...,3:],tex_map[...,3:],tex_map[...,3:]], dim=3)))
                txt_img_orig = ( 1 + txt_img_orig )/2

                it_size = int(deformed_mesh.verts_packed().shape[0]/len(vertices))
                deformed_meshes = [deformed_mesh.verts_packed()[idx : idx + it_size].detach() for idx in range(0, len(deformed_mesh.verts_packed()), it_size)]
                render_mesh = SMPL_Mesh(deformed_meshes, faces, txt_img_orig, device)
                images_predicted = my_render_hard(render_mesh.to(device), trans, global_mat, S)

                predicted_rgb_orig = images_predicted[..., :3]

                ## Create residual texture
                with torch.no_grad():
                    txt_img, _ = model_tex_res(torch.cat([txt_img_orig, tex_map[...,3:]], dim=3))
                txt_img = ( 1 + txt_img )/2

                ## CREATE MESH with texture 
                deformed_meshes = [deformed_mesh.verts_packed()[idx : idx + it_size].detach() for idx in range(0, len(deformed_mesh.verts_packed()), it_size)]
                render_mesh = SMPL_Mesh(deformed_meshes, faces, txt_img, device)

                
                    
                ## RENDER
                images_predicted = my_render_hard(render_mesh.to(device), trans, global_mat, S)
                predicted_seg = images_predicted[..., 3:]
                predicted_seg = (torch.where(predicted_seg < 0.001, predicted_seg, torch.ones_like(predicted_seg))).to(device).detach()

                for image in predicted_seg: 
                    mask_list.append(image.cpu().numpy()*255)

                predicted_rgb = images_predicted[..., :3]

                if args.save_mesh_texture:
                                        
                    my_transform = Transform3d(device=device, matrix=torch.transpose(global_mat.view(-1, 4, 4).to(device), 1, 2)).translate(trans[:,0],trans[:,1], trans[:,2]).scale(S)  
                    verts_camera = my_transform.transform_points(render_mesh.verts_padded())
                    mesh2render = render_mesh.update_padded(new_verts_padded=verts_camera)                    


                for i in range(images_predicted.shape[0]):
                    predicted_path_rgb = os.path.join(output_path_rgb_pred, "TEST{:05d}.jpg".format(step))
                    predicted_path_mask = os.path.join(output_path_mask_pred, "TEST{:05d}.jpg".format(step))
                    #predicted_path_sil = os.path.join(output_path_sil_pred, "TEST{:05d}.jpg".format(step))
                    #print("Writing {}".format(predicted_path_rgb))
                    #print("Writing {}".format(predicted_path_sil))

                    pred_rgb = predicted_rgb.cpu().detach().numpy()[i,:,:,-1::-1]*255
                    cv2.imwrite(predicted_path_rgb, pred_rgb)
                    cv2.imwrite(predicted_path_mask, mask_list[i])
                    #cv2.imwrite(predicted_path_sil, predicted_silhouette.cpu().detach().numpy()[0,:,:,:]*255)

                  
                    if args.save_mesh_texture:

                        #import pdb
                        #pdb.set_trace()

                        vt = np.load(config.SMPL_VT)
                        ft = np.load(config.SMPL_FT)
                                     
                        out_vert,out_faces = mesh2render.get_mesh_verts_faces(i)
                        final_tex = torch.cat([tex_map[...,3:],tex_map[...,3:], tex_map[...,3:]], dim=3)*txt_img + (torch.ones_like(torch.cat([tex_map[...,3:],tex_map[...,3:], tex_map[...,3:]], dim=3)) - torch.cat([tex_map[...,3:],tex_map[...,3:], tex_map[...,3:]], dim=3))*txt_img_orig   

                        write_obj(output_path_mesh_pred + "/model_%05d"%step,out_vert,out_faces,vt,ft,final_tex,i)
                 
                        #cv2.imwrite(output_path_texture_pred + "/coarse_tex_%04d"%step + ".jpg",txt_img_orig.cpu().detach().numpy()[0,:,:,-1::-1]*255)

                        

                        #cv2.imwrite(output_path_texture_pred  + "/final_%05d_in"%step + ".jpg", final_tex.cpu().detach().numpy()[i,:,:,-1::-1]*255)
                   

                        #cv2.imwrite(output_path_texture_pred  + "/vm_%09d_in"%step + ".jpg", tex_map.cpu().detach().numpy()[i,:,:,3:]*255)



                    step += 1
                mask_list = []

        ## Make video
        #os.system('ffmpeg -hide_banner -loglevel panic -framerate 30 -i {}/TEST%05d.jpg {}/{}.mp4'.format(output_path_rgb_pred, os.path.join(output_path, args.source), movement))

 
if __name__ == '__main__':
    main()
