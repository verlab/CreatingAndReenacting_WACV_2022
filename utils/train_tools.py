import cv2
from models.mesh import SMPL_Mesh,TEX_Mesh
from models.render import Render_SMPL,Render_TEX
from utils import dataloader

from pytorch3d.structures import Meshes
from pytorch3d.transforms import Transform3d 


def test(dataloader, f, img_shape, device, model, model_tex, summary):
    print("Testing trained model on cone and box videos...")
    movements = ['bruno', 'box', 'cone']
    dataloaders = dataloader.get_dataloaders(args, phase = "test", movements=movements)

    video = []

    model.eval()
    model_tex.eval()
    my_render_hard = Render_SMPL(f, img_shape, 1000, device, "hard").to(device)

    for idx, (vertices, trans, global_mat) in enumerate(tqdm(dataloaders['test'])):
        
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