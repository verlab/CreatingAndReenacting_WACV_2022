from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.io import load_objs_as_meshes, save_obj

from pytorch3d.utils import ico_sphere
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)
from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    SoftSilhouetteShader,
    SoftPhongShader,
    TexturesVertex
)

import numpy as np

from skimage.io import imread

import sys
import pdb

import os

sys.path.append("..")
from utils.dataloader import RetargetingDataset, get_dataloaders

import argparse
import yaml

from models.meshNet import *

from tqdm import tqdm

# Losses to smooth / regularize the mesh shape
def update_mesh_shape_prior_losses(mesh, loss):
    # and (b) the edge length of the predicted mesh
    loss["edge"] = mesh_edge_loss(mesh)
    
    # mesh normal consistency
    loss["normal"] = mesh_normal_consistency(mesh)
    
    # mesh laplacian smoothing
    loss["laplacian"] = mesh_laplacian_smoothing(mesh, method="uniform")

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

    ## TESTE DO MODELO:
    DATA_DIR = "./data/"
    obj_filename = os.path.join(DATA_DIR, "cow_mesh/cow.obj")

    # Load obj file
    mesh = load_objs_as_meshes([obj_filename], device=device)

    # We scale normalize and center the target mesh to fit in a sphere of radius 1 
    # centered at (0,0,0). (scale, center) will be used to bring the predicted mesh 
    # to its original center and scale.  Note that normalizing the target mesh, 
    # speeds up the optimization but is not necessary!
    verts = mesh.verts_packed()
    N = verts.shape[0]
    center = verts.mean(0)
    scale = max((verts - center).abs().max(0)[0])
    mesh.offset_verts_(-center.expand(N, 3))
    mesh.scale_verts_((1.0 / float(scale)))

    ## DATASET CREATION
    num_views = 20

    # Get a batch of viewing angles. 
    elev = torch.linspace(0, 360, num_views)
    azim = torch.linspace(-180, 180, num_views)

    # Place a point light in front of the object. As mentioned above, the front of 
    # the cow is facing the -z direction. 
    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

    # Initialize an OpenGL perspective camera that represents a batch of different 
    # viewing angles. All the cameras helper methods support mixed type inputs and 
    # broadcasting. So we can view the camera from the a distance of dist=2.7, and 
    # then specify elevation and azimuth angles for each viewpoint as tensors. 
    R, T = look_at_view_transform(dist=2.7, elev=elev, azim=azim)
    cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T)

    # We arbitrarily choose one particular view that will be used to visualize 
    # results
    camera = OpenGLPerspectiveCameras(device=device, R=R[None, 1, ...], 
                                    T=T[None, 1, ...]) 

    # Define the settings for rasterization and shading. Here we set the output 
    # image to be of size 128X128. As we are rendering images for visualization 
    # purposes only we will set faces_per_pixel=1 and blur_radius=0.0. Refer to 
    # rasterize_meshes.py for explanations of these parameters.  We also leave 
    # bin_size and max_faces_per_bin to their default values of None, which sets 
    # their values using huristics and ensures that the faster coarse-to-fine 
    # rasterization method is used.  Refer to docs/notes/renderer.md for an 
    # explanation of the difference between naive and coarse-to-fine rasterization. 
    raster_settings = RasterizationSettings(
        image_size=128, 
        blur_radius=0.0, 
        faces_per_pixel=1, 
    )

    # Create a phong renderer by composing a rasterizer and a shader. The textured 
    # phong shader will interpolate the texture uv coordinates for each vertex, 
    # sample from a texture image and apply the Phong lighting model
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=camera, 
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device, 
            cameras=camera,
            lights=lights
        )
    )

    # Create a batch of meshes by repeating the cow mesh and associated textures. 
    # Meshes has a useful `extend` method which allows us do this very easily. 
    # This also extends the textures. 
    meshes = mesh.extend(num_views)

    # Render the cow mesh from each viewing angle
    target_images = renderer(meshes, cameras=cameras, lights=lights)

    # Our multi-view cow dataset will be represented by these 2 lists of tensors,
    # each of length num_views.
    target_rgb = [target_images[i, ..., :3] for i in range(num_views)]
    target_cameras = [OpenGLPerspectiveCameras(device=device, R=R[None, i, ...], 
                                            T=T[None, i, ...]) for i in range(num_views)]

    sigma = 1e-4
    raster_settings_silhouette = RasterizationSettings(
        image_size=128, 
        blur_radius=np.log(1. / 1e-4 - 1.)*sigma, 
        faces_per_pixel=50, 
    )

    # Silhouette renderer 
    renderer_silhouette = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=camera, 
            raster_settings=raster_settings_silhouette
        ),
        shader=SoftSilhouetteShader()
    )

    # We initialize the source shape to be a sphere of radius 1.  
    src_mesh = ico_sphere(4, device)

    # Rasterization settings for differentiable rendering, where the blur_radius
    # initialization is based on Liu et al, 'Soft Rasterizer: A Differentiable 
    # Renderer for Image-based 3D Reasoning', ICCV 2019
    sigma = 1e-4
    raster_settings_soft = RasterizationSettings(
        image_size=128, 
        blur_radius=np.log(1. / 1e-4 - 1.)*sigma, 
        faces_per_pixel=50, 
    )

    # Differentiable soft renderer using per vertex RGB colors for texture
    renderer_textured = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=camera, 
            raster_settings=raster_settings_soft
        ),
        shader=SoftPhongShader(device=device, 
            cameras=camera,
            lights=lights)
    )

    # Render silhouette images.  The 3rd channel of the rendering output is 
    # the alpha/silhouette channel
    silhouette_images = renderer_silhouette(meshes, cameras=cameras, lights=lights)
    target_silhouette = [silhouette_images[i, ..., 3] for i in range(num_views)]

    # Number of views to optimize over in each SGD iteration
    num_views_per_iteration = 2
    # Number of optimization steps
    Niter = 2000
    # Plot period for the losses

    # Optimize using rendered RGB image loss, rendered silhouette image loss, mesh 
    # edge loss, mesh normal consistency, and mesh laplacian smoothing
    losses = {"rgb": {"weight": 1.0, "values": []},
            "silhouette": {"weight": 1.0, "values": []},
            "edge": {"weight": 1.0, "values": []},
            "normal": {"weight": 0.01, "values": []},
            "laplacian": {"weight": 1.0, "values": []},
            }

    # We will learn to deform the source mesh by offsetting its vertices
    # The shape of the deform parameters is equal to the total number of vertices in 
    # src_mesh
    verts_shape = src_mesh.verts_packed().shape
    deform_verts = torch.full(verts_shape, 0.0, device=device, requires_grad=True)

    # We will also learn per vertex colors for our sphere mesh that define texture 
    # of the mesh
    sphere_verts_rgb = torch.full([1, verts_shape[0], 3], 0.5, device=device, requires_grad=True)

    # The optimizer
    ## TESTE
    #pdb.set_trace()
    #all_params = list(model.parameters())
    #all_params.append(sphere_verts_rgb)
    #optimizer = torch.optim.SGD(all_params, lr=0.05, momentum=0.9)
    #optimizer = torch.optim.Adam(all_params, lr=0.005, betas=(0.5, 0.999))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, betas=(0.5, 0.999))
    optimizer_exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1, last_epoch=-1)

    #optimizer = torch.optim.SGD([deform_verts, sphere_verts_rgb], lr=1.0, momentum=0.9)
    
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
    texture_optm = torch.optim.SGD([sphere_verts_rgb], lr=0.5, momentum=0.9)
    texture_optm = torch.optim.Adam([sphere_verts_rgb], lr=0.005, betas=(0.5, 0.999))

    loop = tqdm(range(Niter))

    for i in loop:
        # Initialize optimizer
        optimizer.zero_grad()
        texture_optm.zero_grad()
        
        # Deform the mesh
        subdivide = False
        new_src_mesh = model(src_mesh, subdivide)[-1]

        #new_src_mesh = src_mesh.offset_verts(deform_verts)
        
        # Add per vertex colors to texture the mesh
        new_src_mesh.textures = TexturesVertex(verts_features=sphere_verts_rgb) 
        
        # Losses to smooth /regularize the mesh shape
        loss = {k: torch.tensor(0.0, device=device) for k in losses}
        update_mesh_shape_prior_losses(new_src_mesh, loss)
        
        # Randomly select two views to optimize over in this iteration.  Compared
        # to using just one view, this helps resolve ambiguities between updating
        # mesh shape vs. updating mesh texture
        for j in np.random.permutation(num_views).tolist()[:num_views_per_iteration]:
            images_predicted = renderer_textured(new_src_mesh, cameras=target_cameras[j], lights=lights)

            # Squared L2 distance between the predicted silhouette and the target 
            # silhouette from our dataset
            predicted_silhouette = images_predicted[..., 3]
            loss_silhouette = ((predicted_silhouette - target_silhouette[j]) ** 2).mean()
            loss["silhouette"] += loss_silhouette / num_views_per_iteration
            
            # Squared L2 distance between the predicted RGB image and the target 
            # image from our dataset
            predicted_rgb = images_predicted[..., :3]
            loss_rgb = ((predicted_rgb - target_rgb[j]) ** 2).mean()
            loss["rgb"] += loss_rgb / num_views_per_iteration
        
        # Weighted sum of the losses
        sum_loss = torch.tensor(0.0, device=device)
        for k, l in loss.items():
            sum_loss += l * losses[k]["weight"]
            losses[k]["values"].append(l)
        
        # Print the losses
        loop.set_description("total_loss = %.6f" % sum_loss)
        
        # Optimization step
        sum_loss.backward()
        optimizer.step()
        texture_optm.step()

    # Fetch the verts and faces of the final predicted mesh
    final_verts, final_faces = new_src_mesh[-1].get_mesh_verts_faces(0)
    #final_verts, final_faces = new_src_mesh.get_mesh_verts_faces(0)

    # Scale normalize back to the original target size
    final_verts = final_verts * scale + center

    # Store the predicted mesh using save_obj
    final_obj = os.path.join('./', 'final_model.obj')
    save_obj(final_obj, final_verts, final_faces)

if __name__ == '__main__':
    main()