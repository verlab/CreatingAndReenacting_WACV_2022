# coding=utf-8
# Copyright 2020 The Google AI Perception Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Test code for running visualizer."""
import os

from absl import app
from absl import flags
from aist_plusplus.loader import AISTDataset
#from aist_plusplus.visualizer import plot_on_video
from smplx import SMPL
import torch

import sys
sys.path.append("..")
from models.smpl import SMPL,load_smpl
import config
import os.path as osp
import pickle

import torch.nn.functional as F
from smplx.lbs import (batch_rodrigues, transform_mat)
import cv2
import numpy as np
from utils.mesh_tools import write_obj

from models.render import Render_SMPL
from models.mesh import SMPL_Mesh
from models.smpl import SMPL,load_smpl
from utils.mesh_tools import write_obj

import argparse
import os
import sys
import urllib.request
from functools import partial
import random
SOURCE_URL = 'https://aistdancedb.ongaaccel.jp/v1.0.0/video/10M/'
LIST_URL = 'https://storage.googleapis.com/aist_plusplus_public/20121228/video_list.txt'

def _download(video_url, download_folder):
  save_path = os.path.join(download_folder, os.path.basename(video_url))
  urllib.request.urlretrieve(video_url, save_path)


import pdb

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'anno_dir',
    '/homeLocal/thiagoluange/AIST/aist_plusplus_final/',
    'input local dictionary for AIST++ annotations.')
flags.DEFINE_string(
    'video_dir',
    '/media/thiagoluange/SAMSUNG/AIST/d19/',
    'input local dictionary for AIST Dance Videos.')

flags.DEFINE_string(
    'save_video_dir',
    '/homeLocal/thiagoluange/AIST/aist_plusplus_final/',
    'input local dictionary for save AIST Dance Videos.')
flags.DEFINE_string(
    'smpl_dir',
    '/code/data/smpl/',
    'input local dictionary that stores SMPL data.')
flags.DEFINE_string(
    'video_name',
    'gHO_sFM_c02_d19_mHO2_ch07',
    'input video name to be visualized.')

flags.DEFINE_string(
    'target_name',
    'd04',
    'target to be download')

flags.DEFINE_integer('render', 0, 'render vis pose')
flags.DEFINE_integer('download', 1, 'download videos')

flags.DEFINE_string(
    'save_dir',
    '/srv/storage/datasets/thiagoluange/AIST/',
    'output local dictionary that stores AIST++ visualization.')


class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

def load_smpl_local():    

    gender="male"
  
    model_path = config.SMPL_MODEL_DIR

    if osp.isdir(model_path):
        model_fn = 'SMPL_{}.{ext}'.format(gender.upper(), ext='pkl')
        smpl_path = os.path.join(model_path, model_fn)
    else:
        smpl_path = model_path
        assert osp.exists(smpl_path), 'Path {} does not exist!'.format(smpl_path)

    with open(smpl_path, 'rb') as smpl_file:
        model = pickle.load(smpl_file,encoding='latin1')
        data_struct = Struct(**model)

    my_smpl = SMPL("",data_struct=data_struct ,batch_size=1,create_transl=False,gender=gender)
   
    return my_smpl

def make_R(rvec):
    out = np.zeros((4,4))
    rotmat, _ = cv2.Rodrigues(rvec)
    out[:3,:3] = rotmat
    out[3, 3] = 1
    return out

def make_T(tvec):
    out = np.zeros((4,4))
    out[0, 0] = 1
    out[1, 1] = 1
    out[2, 2] = 1
    out[3, 3] = 1
    out[:3, 3] = tvec.flatten()

    return out
def make_S(s):
    out = np.zeros((4,4))
    out[0, 0] = s
    out[1, 1] = s
    out[2, 2] = s
    out[3, 3] = 1
    

    return out

def main(_):

  print ("load libs ok")

  if torch.cuda.is_available():
      device = torch.device("cuda:0")
      torch.cuda.set_device(device)
  else:
      device = torch.device("cpu")

  print ("Downloading phase")

  os.makedirs(FLAGS.save_video_dir + "/videos_" + FLAGS.target_name + "/", exist_ok=True)
  
  os.makedirs(FLAGS.save_dir + "/" + FLAGS.target_name + "/vis/", exist_ok=True)
  os.makedirs(FLAGS.save_dir + "/" + FLAGS.target_name + "/frames/", exist_ok=True)
  os.makedirs(FLAGS.save_dir + "/" + FLAGS.target_name + "/test_pose_new/", exist_ok=True)


  seq_names = urllib.request.urlopen(LIST_URL)

  
  seq_names = [seq_name.strip().decode('utf-8') for seq_name in seq_names]
  seq_names = [seq_name for seq_name in seq_names if ((FLAGS.target_name in seq_name)) ]
  seq_names = [seq_name for seq_name in seq_names if not (("c09" in seq_name)) ]
  #seq_names = [seq_name for seq_name in seq_names if not (("mBR" in seq_name)) ]

  with open(FLAGS.anno_dir + "/ignore_list.txt") as f:
      ignore_videos = f.readlines()
        
      ignore_videos = [seq_name[:-1] for seq_name in ignore_videos if len(seq_name) > 2]

      videos_to_ig = []

      for ig in ignore_videos:
         ig = ig.split("cAll")          
         for c in ["c01","c02","c03","c04","c05","c06","c07","c08"]:
             videos_to_ig.append(ig[0] + c + ig[1])
          
      for ig in videos_to_ig :          
          seq_names = [seq_name for seq_name in seq_names if not ((ig in seq_name)) ]


  if FLAGS.download == 1:
 
      mypath = FLAGS.save_video_dir + "/videos_" + FLAGS.target_name + "/"

      onlyfiles = [f for f in os.listdir(mypath) if os.path.isfile(mypath + "/" + f)]

      print ("Videos already used:")
      print (onlyfiles)
      print (len(seq_names))

      for ig in onlyfiles:
          ig = ig.split(".")[0]         
          seq_names = [seq_name for seq_name in seq_names if not ((ig in seq_name)) ]

  else:
      mypath = FLAGS.save_video_dir + "/videos_" + FLAGS.target_name + "/"

      onlyfiles = [f for f in os.listdir(mypath) if os.path.isfile(mypath + "/" + f)]
      seq_names = []

      for ig in onlyfiles:
          ig = ig.split(".")[0]         
          seq_names.append(ig)

  print (len(seq_names))

  random.shuffle(seq_names)

  print (len(seq_names))
  video_urls = [
      os.path.join(SOURCE_URL, seq_name + '.mp4') for seq_name in seq_names]


  aist_dataset = AISTDataset(FLAGS.anno_dir)

  #download_func = partial(_download, download_folder=args.download_folder)
  #pool = multiprocessing.Pool(processes=args.num_processes)

  
  mypath =  FLAGS.save_dir + "/" + FLAGS.target_name + "/frames/"
  onlyfiles = [f for f in os.listdir(mypath) if os.path.isfile(mypath + "/" + f)]

  extracted_frames = len(onlyfiles)

  print ("extracted frames")
  print(extracted_frames)

  my_dict_pkl = {'pose':np.zeros(72),'betas':np.zeros(10),'model_type': 1,'f':1500,'img_shape': (1080, 1920, 3),'rt': np.zeros(3),'trans':[],'t':np.zeros(3),'global_mat':np.zeros((4,4))}

  for i, _ in enumerate(video_urls):

      if extracted_frames > 7000:
          exit()

      if FLAGS.download == 1:

          print ('\rdownloading %d / %d' % (i + 1, len(video_urls)))
          print (seq_names[i])
          _download(video_urls[i], FLAGS.save_video_dir + "/videos_" + FLAGS.target_name + "/")
          print ("Done Download")

      video_path = os.path.join(FLAGS.save_video_dir + "/videos_" + FLAGS.target_name + "/", f'{seq_names[i]}.mp4')
      seq_name, view = AISTDataset.get_seq_name(seq_names[i])
      view_idx = AISTDataset.VIEWS.index(view)
     
      
      file_path = os.path.join(aist_dataset.motion_dir, f'{seq_name}.pkl')
      smpl_poses, smpl_scaling, smpl_trans = AISTDataset.load_motion(aist_dataset.motion_dir, seq_name)
      env_name = aist_dataset.mapping_seq2env[seq_name]
      cgroup = AISTDataset.load_camera_group(aist_dataset.camera_dir, env_name)
      cap = cv2.VideoCapture(video_path)

      for camera in cgroup:
          if camera['name'] == view:
              camera_now = camera
  
      rotation = camera_now['rotation']
 
  
      my_mat_R = make_R(np.array(rotation))

      my_camera_trans = np.array(camera_now['translation'])/smpl_scaling[0]


      img_shape = camera_now['size']
  
      f = (camera_now['matrix'][0][0] + camera_now['matrix'][1][1])/2.0
  

      my_smpl = load_smpl_local()
      
      txt_img = cv2.resize(cv2.imread(config.TEX_MAP, cv2.IMREAD_UNCHANGED),(256,256))
  
      txt_img =  torch.from_numpy(txt_img[:,:,::-1].astype('float64')/255.0).to(device)
  
      my_render_hard = Render_SMPL(f, img_shape, 200, device, "hard",eye=[[0,0,0]],at=[[0,0,-1]], up=[[0, 1, 0]]).to(device)
  

      faces_mesh = torch.from_numpy(np.load(config.SMPL_FACES)).to(device)

      with open(config.SMPL_IN_VERT) as f:
          list_in_verts = f.readlines()
          in_verts = [int(x) for x in list_in_verts] 

  
      print ("Extracting frames")

      for id_pose in range(0,smpl_poses.shape[0],2):
          print ("process pose " + str(id_pose))
          cap.set(1, id_pose)
          res, frame = cap.read()

          if(frame is None):
              continue

          frame = cv2.flip(frame, 1)
          cv2.imwrite(FLAGS.save_dir + "/" + FLAGS.target_name + "/frames/" +  "img%08d.jpg"%extracted_frames,frame)

          frame = frame[:,420:1500,:]
          frame = cv2.resize(frame,(200,200))
          
          global_orient = torch.Tensor(smpl_poses[id_pose,:3]).unsqueeze(0)

          body_pose     = torch.Tensor(smpl_poses[id_pose,3:72]).unsqueeze(0)
          smpl_output = my_smpl(global_orient=torch.Tensor([[0.0,0.0,0.0]]),body_pose=body_pose)

      
          joints_homogen = F.pad(smpl_output.joints[0,0,:], [0, 1])
          transform = transform_mat(batch_rodrigues(global_orient),smpl_output.joints[0,0,:].view(-1,3,1))
          rel_transforms = transform - torch.transpose(F.pad(torch.matmul(transform, joints_homogen), [0, 0, 3, 0]),0,1)
      
          

          my_mat_T = make_T(np.array([smpl_trans[id_pose,0]/smpl_scaling[0],smpl_trans[id_pose,1]/smpl_scaling[0],smpl_trans[id_pose,2]/smpl_scaling[0]]))

          #pdb.set_trace()
          global_mat = torch.matmul(torch.Tensor(my_mat_T),rel_transforms)
          global_mat = torch.matmul(torch.Tensor(my_mat_R),global_mat)
 
          trans = my_camera_trans.reshape((-1,3))
            

          if FLAGS.render == 1:

              tex_maps = []
              for i in range(1):
                  tex_maps.append(txt_img)

              tex_map = torch.stack(tex_maps)
              my_mesh = SMPL_Mesh(smpl_output.vertices.squeeze(0)[in_verts,:].view(1,-1,3), faces_mesh.view(1,-1,3),tex_map,device)
              images_predicted = my_render_hard(my_mesh.to(device), trans,global_mat)
      

      
              my_img = images_predicted[..., :3].cpu().detach().numpy()[0,:,:,-1::-1]*255

              mask = images_predicted[..., 3:].cpu().detach().numpy()[0,:,:,0]

              mask = np.where(mask > 0.0,np.ones_like(mask),np.zeros_like(mask))
   
              mask = np.stack([mask,mask,mask],axis=2)

              my_img = mask*(frame*0.5 + my_img*0.5) + (np.ones_like(mask) - mask)*frame

              cv2.imwrite(FLAGS.save_dir + "/" + FLAGS.target_name + "/vis/" +  "saida%08d.jpg"%extracted_frames,my_img)

          my_dict_pkl['pose'] = smpl_poses[id_pose,:]
          my_dict_pkl['f'] = (camera_now['matrix'][0][0] + camera_now['matrix'][1][1])/2.0
          my_dict_pkl['trans'] = trans.tolist()
          my_dict_pkl['global_mat'] = global_mat.cpu().detach().numpy()

          with open(FLAGS.save_dir + "/" + FLAGS.target_name + "/test_pose_new/" +  "motion%08d.pkl"%extracted_frames, "wb" ) as f:
              pickle.dump(my_dict_pkl, f )

          

          extracted_frames = extracted_frames + 1


          if extracted_frames > 7000:
              exit()
      

      

 


if __name__ == '__main__':
  app.run(main)
tex_map
