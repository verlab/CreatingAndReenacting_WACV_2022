import torch
import torch.nn as nn
from pytorch3d.ops import GraphConv, SubdivideMeshes, vert_align
from torch.nn import functional as F

import config
import numpy as np

class MeshRefinementStage(nn.Module):
    def __init__(self, vert_feat_dim, hidden_dim, stage_depth, device, batch_size, gconv_init="normal"):
        """
        Args:
          vert_feat_dim: Dimension of vert_feats we will receive from the
                        previous stage; can be 0
          hidden_dim: Output dimension for graph-conv layers
          stage_depth: Number of graph-conv layers to use
          gconv_init: How to initialize graph-conv layers
        """
        super(MeshRefinementStage, self).__init__()
        self.batch_size = batch_size
        self.anchor_v = (torch.from_numpy(np.load(config.ANCHOR_V)['arr_0'])).type(torch.FloatTensor).to(device)
        with open(config.SMPL_IN_VERT) as f:
            list_in_verts = f.readlines()
            in_verts = [int(x) for x in list_in_verts] 

        with open(config._MODEL_MAP_FNAME,"r") as f: 
            pose_partes = f.readlines()
            pose_partes = np.array([int(float(x)) for x in pose_partes])
            pose_partes = pose_partes[in_verts]

        pose_partes_id = [ [] for i in range(14)]

        for i in range(pose_partes.shape[0]): 
            pose_partes_id[pose_partes[i] -1].append(i)    


        self.pose_partes_id = pose_partes_id

        limit_deformation = [0.10]*14

        #cabeca
        limit_deformation[0] = 0.04  
        #tronco
        limit_deformation[1] = 0.06  
        #antebraco
        limit_deformation[2] = 0.04 
        limit_deformation[3] = 0.04 
        #braco 
        limit_deformation[4] = 0.02
        limit_deformation[5] = 0.02
        #coxa
        limit_deformation[6] = 0.04
        limit_deformation[7] = 0.04
        #panturrilha
        limit_deformation[8] = 0.03
        limit_deformation[9] = 0.03
        #pes
        limit_deformation[10] = 0.02
        limit_deformation[11] = 0.02
        #maos
        limit_deformation[12] = 0.01
        limit_deformation[13] = 0.01

        self.limit_deformation = limit_deformation

        #self.anchor_v = torch.cat([self.anchor_v] * self.batch_size)

        # deform layer
        self.verts_offset = nn.Linear(hidden_dim + 3, 3)
        self.device = device
        self.ft_size = 6596.0
        # graph convs
        self.gconvs = nn.ModuleList()
        for i in range(stage_depth):
            if i == 0:
                input_dim = vert_feat_dim + 3
            else:
                input_dim = hidden_dim + 3
            gconv = GraphConv(input_dim, hidden_dim, init=gconv_init, directed=False)
            self.gconvs.append(gconv)

        # initialization
        nn.init.zeros_(self.verts_offset.weight)
        nn.init.constant_(self.verts_offset.bias, 0)

    def forward(self, meshes, vert_feats=None):

        #vert_pos_packed = meshes.verts_packed()
        vert_pos_packed = meshes.verts_normals_packed()

        first_layer_feats = [vert_pos_packed]
        if vert_feats is not None:
            first_layer_feats.append(vert_feats)
        vert_feats = torch.cat(first_layer_feats, dim=1)

        for gconv in self.gconvs:
            vert_feats_nopos = F.relu(gconv(vert_feats.to(self.device), meshes.edges_packed().to(self.device))).to(self.device)
            vert_feats = torch.cat([vert_feats_nopos, vert_pos_packed.to(self.device)], dim=1)

        # refine
        deform = torch.tanh(self.verts_offset(vert_feats)).to(self.device)

        meshes.verts_packed().to(self.device)
        meshes.faces_packed().to(self.device)
        _size = deform.shape[0]/self.ft_size

        # clamp limiar

        clamp_deformed = torch.zeros_like(deform)

        for i in range(int(_size)):
             for j in range(len(self.pose_partes_id)):
                 id_part = [ (x + i*6596) for x in self.pose_partes_id[j]]
                 clamp_deformed[id_part,:] = torch.clamp(deform[id_part,:], min=-1.0*self.limit_deformation[j], max=self.limit_deformation[j])

        anchor_v = torch.cat([self.anchor_v] * int(_size))        

        
        meshes = meshes.offset_verts(clamp_deformed*anchor_v) ## Tirar o block do pé
        
        return meshes, vert_feats_nopos

class MeshRefinementHead(nn.Module):
    def __init__(self, cfg):
        super(MeshRefinementHead, self).__init__()
        # fmt: off
        self.num_stages = cfg['num_stages']
        hidden_dim      = cfg['graph_conv_dim']
        stage_depth     = cfg['num_graph_convs']
        graph_conv_init = cfg['graph_conv_init']
        self.device = cfg["device"]
        self.batch_size = cfg["batch_size"]
        # fmt: on

        self.stages = nn.ModuleList()
        for i in range(self.num_stages):
            vert_feat_dim = 0 if i == 0 else hidden_dim
            stage = MeshRefinementStage(
                vert_feat_dim, hidden_dim, stage_depth, self.device, self.batch_size, gconv_init=graph_conv_init
            )
            self.stages.append(stage)

    def forward(self, meshes, subdivide=False):
        ## Creio que subdivide seja para aumentar a mesh e 'refiná-la'.
        """
        Args:
          img_feats (tensor): Tensor of shape (N, C, H, W) giving image features,
                              or a list of such tensors.
          meshes (Meshes): Meshes class of N meshes
          P (tensor): Tensor of shape (N, 4, 4) giving projection matrix to be applied
                      to vertex positions before vert-align. If None, don't project verts.
          subdivide (bool): Flag whether to subdivice the mesh after refinement
        Returns:
          output_meshes (list of Meshes): A list with S Meshes, where S is the
                                          number of refinement stages
        """
        output_meshes = None
        vert_feats = None
        for i, stage in enumerate(self.stages):
            meshes, vert_feats = stage(meshes.to(self.device), vert_feats)
            meshes = meshes.to(self.device)
            vert_feats = vert_feats.to(self.device)
            if subdivide and i < self.num_stages - 1:
                subdivide = SubdivideMeshes()
                meshes, vert_feats = subdivide(meshes, feats=vert_feats)
        return meshes
