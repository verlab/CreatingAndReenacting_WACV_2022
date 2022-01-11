from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes, join_meshes_as_batch
from pytorch3d.utils import ico_sphere
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)

import sys
import pdb

import os

sys.path.append("..")
from utils.dataloader import RetargetingDataset, get_dataloaders

import argparse
import yaml

from models.meshNet import *

from tqdm import tqdm

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", default="S1") ## Setting default args to avoid writting them
    parser.add_argument("-g", "--gender", default="female")
    parser.add_argument("-p", "--person", default="P0")
    parser.add_argument("-d", "--dataset_path", default = "/srv/storage/datasets/thiagoluange/dd_dataset")
    parser.add_argument("-t", "--test_path", default = "/srv/storage/datasets/thiagoluange/dd_dataset")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"RUNNING ON: {device}")

    with open("../models/model_cfg.yaml", 'r') as f:
        model_cfgs = yaml.safe_load(f)

    model = MeshRefinementHead(model_cfgs).to(device)

    '''phase = 'train'
    dataloaders = get_dataloaders(args, phase)
    for idx, (vertices, faces, seg, img, render_params) in enumerate(dataloaders[phase]):
        print(idx)
        pdb.set_trace()'''

    ## TESTE DO MODELO: ESFERA --> GOLFINHO

    trg_obj = os.path.join('../../dolphin.obj')
    # We read the target 3D model using load_obj
    verts, faces, aux = load_obj(trg_obj)
    faces_idx = faces.verts_idx.to(device)
    verts = verts.to(device)

    center = verts.mean(0)
    verts = verts - center
    scale = max(verts.abs().max(0)[0])
    verts = verts / scale

    trg_mesh = Meshes(verts=[verts], faces=[faces_idx])

    src_mesh = ico_sphere(4, device)

    #####
    deform_verts = torch.full(src_mesh.verts_packed().shape, 0.0, device=device, requires_grad=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)

    # Number of optimization steps
    Niter = 2000
    # Weight for the chamfer loss
    w_chamfer = 1.0 
    # Weight for mesh edge loss
    w_edge = 1.0 
    # Weight for mesh normal consistency
    w_normal = 0.01 
    # Weight for mesh laplacian smoothing
    w_laplacian = 0.1 
    # Plot period for the losses
    plot_period = 250
    loop = tqdm(range(Niter))

    chamfer_losses = []
    laplacian_losses = []
    edge_losses = []
    normal_losses = []

    for i in loop:
        # Initialize optimizer
        optimizer.zero_grad()
        
        # Deform the mesh
        subdivide = False
        new_src_mesh = model(src_mesh, subdivide)[-1]

        #new_src_mesh = src_mesh.offset_verts(deform_verts)
        
        # We sample 5k points from the surface of each mesh 
        sample_trg = sample_points_from_meshes(trg_mesh, 5000)
        sample_src = sample_points_from_meshes(new_src_mesh, 5000)
        
        # We compare the two sets of pointclouds by computing (a) the chamfer loss
        loss_chamfer, _ = chamfer_distance(sample_trg, sample_src)
        
        # and (b) the edge length of the predicted mesh
        loss_edge = mesh_edge_loss(new_src_mesh)
        
        # mesh normal consistency
        loss_normal = mesh_normal_consistency(new_src_mesh)
        
        # mesh laplacian smoothing
        loss_laplacian = mesh_laplacian_smoothing(new_src_mesh, method="uniform")
        
        # Weighted sum of the losses
        loss = loss_chamfer * w_chamfer + loss_edge * w_edge + loss_normal * w_normal + loss_laplacian * w_laplacian
        
        # Print the losses
        loop.set_description('total_loss = %.6f' % loss)
        
        # Save the losses for plotting
        chamfer_losses.append(loss_chamfer)
        edge_losses.append(loss_edge)
        normal_losses.append(loss_normal)
        laplacian_losses.append(loss_laplacian)
            
        # Optimization step
        loss.backward()
        optimizer.step()

if __name__ == '__main__':
    main()