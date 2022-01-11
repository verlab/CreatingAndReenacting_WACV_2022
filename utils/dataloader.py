import os, sys, inspect
#fast fix for import from parent folder. right thing to do would be organize the folders.
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
import torch
import torch
import numpy as np
import pickle
import smplx
from models.smpl import SMPL,load_smpl
from utils.mesh_tools import write_obj
import config
import constants
import torch.utils.data
import re
from torchvision import transforms
from tqdm import tqdm

from time import time
import torch.nn.functional as F

from PIL import Image
import pdb
import argparse
from torchvision.transforms.functional import center_crop, affine
from smplx.lbs import (batch_rodrigues, transform_mat)
import numpy as np
import cv2
import math
from torch.utils.data import WeightedRandomSampler
import config

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    if x > 0:
        phi = np.arctan2(y, x)
    else:
        phi = np.pi + np.arctan2(-1.0*y, -1.0*x)
    return rho, phi

class CenterSquaredCrop(object):

    def __init__(self, output_size = None):
        assert isinstance(output_size, (int, tuple, type(None)))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        elif isinstance(output_size, type(None)):
            self.output_size = None
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image):
        w, h = image.size[:2]
        min_dim = h if (h < w) else w
        return center_crop(image, min_dim)

class RetargetingDataset(torch.utils.data.Dataset):
    
    def __init__(self, dataset_path, source, person, gender,render_size_soft,render_size_hard, phase = "train", movement = None, style = "mt",num_rot_classes=8):

        ## NOTE: os arquivos precisam estar ordenados?
        self._source = source
        self._person = person
        self._gender = gender
        self._phase = phase
        self.style = style
        self.movement = movement        
        self._dataset_path = dataset_path
        self._model_type = 1 if self._gender == "male" else 2

        with open(config.SMPL_IN_VERT) as f:
            list_in_verts = f.readlines()
            self.in_verts = [int(x) for x in list_in_verts] 

        if self.movement is not None:
            self.data_root = os.path.join(dataset_path, source, movement, person)
        else:
            self.data_root = os.path.join(dataset_path, "{}{}".format(source, person))

        start_time = time()

        if self._phase == "train":
            self.frames_path = os.path.join(self.data_root, "frames")
            if style == 'mt':
                self.segmentations_path = os.path.join(self.data_root, "segmentations_new")
            else:
                self.segmentations_path = os.path.join(self.data_root, "segmentations")
            self.seg_files = sorted([f for f in os.listdir(self.segmentations_path) if '.png' in f])   
            self.frames_files = sorted(os.listdir(self.frames_path))
        
        self.pkl_path = os.path.join(self.data_root, "test_pose_new")
        self.pkl_files = sorted([f for f in os.listdir(self.pkl_path) if f.endswith(('.pkl'))])
        if(self._phase != "train"):
            self.pkl_files = [f for f in self.pkl_files if "ret.pkl" in f]
        self.samples = []
        
        self.render_size_soft = render_size_soft
        self.render_size_hard = render_size_hard

        #add more transforms as the necessity goes

        self.transform_soft = transforms.Compose(
            [
                CenterSquaredCrop(),
                transforms.Resize(render_size_soft, interpolation=2),
                transforms.ToTensor()
            ]
        )

        self.transform_hard = transforms.Compose(
            [
                CenterSquaredCrop(),
                transforms.Resize(render_size_hard, interpolation=2),
                transforms.ToTensor()
            ]
        )
       
        self.num_rot_classes = num_rot_classes
        self.class_sample_count = np.zeros(num_rot_classes)
        
        if self._phase == "train":        
            assert len(self.pkl_files) == len(self.seg_files), "PKL AND SEG FILES WITH DIFFERENT LENGTHS!"
            assert len(self.pkl_files) == len(self.frames_files), "PKL AND FRAMES FILES WITH DIFFERENT LENGTHS!"

        if(len(self.pkl_files) > 0):
            pkl_full_path = os.path.join(self.pkl_path, self.pkl_files[0])
            _dict = self.load_pickle(pkl_full_path)
            
            self.smpl, self.betas = load_smpl(pkl_full_path, self._model_type)
            self.betas     = torch.from_numpy(self.betas)
            self.f         = _dict['f']
            self.faces     = torch.Tensor(self.smpl.faces.astype(int))
            self.img_shape = _dict['img_shape']
        else:
            print("Received empty pkl file list...")
            exit(1)

        if(self._phase == "train"):
            for pkl_f, seg_f, frame_f in zip(tqdm(self.pkl_files, desc="Processando smpl..."), self.seg_files, self.frames_files):
                sample_dict = {}
                render_params = {}
                pkl_full_path = os.path.join(self.pkl_path, pkl_f)

                _dict = self.load_pickle(pkl_full_path)

                #sample_dict['trans']         = torch.Tensor(_dict['trans']).reshape((3))
                sample_dict['render_params'] = render_params
                sample_dict['vertices'], sample_dict['global_mat'],sample_dict['rot_class'],sample_dict['trans_center'],new_trans,scale_factor,face_posi  = self.expand_smpl(_dict,style,os.path.join(self.frames_path, frame_f))
                sample_dict['scale_factor']         = scale_factor
                sample_dict['trans']         = torch.Tensor(new_trans).reshape((3))
                sample_dict['seg_file']      = os.path.join(self.segmentations_path, seg_f)
                sample_dict['frame']         = os.path.join(self.frames_path, frame_f)
                sample_dict['f']         = _dict['f']
                sample_dict['face_posi']         = face_posi
                self.samples.append(sample_dict)
        else:
            for pkl_f in self.pkl_files:
                sample_dict = {}
                render_params = {}
                pkl_full_path = os.path.join(self.pkl_path, pkl_f)

                _dict = self.load_pickle(pkl_full_path)

                sample_dict['trans']         = torch.Tensor(_dict['trans']).reshape((3))
                sample_dict['render_params'] = render_params
                sample_dict['vertices'], sample_dict['global_mat'],sample_dict['rot_class'],sample_dict['trans_center'],new_trans,scale_factor,face_posi  = self.expand_smpl(_dict,style)
                sample_dict['f']         = _dict['f']       
                     
                sample_dict['face_posi']         = face_posi

                self.samples.append(sample_dict)

        #cont classes
        for si in self.samples:
            self.class_sample_count[si['rot_class']] += 1        
                 
        weight = 1./self.class_sample_count
        self.samples_weight = np.array([weight[t['rot_class']] for t in self.samples])

        end_time = time()
        if phase == "train":
            print(f"TOTAL TIME: {end_time-start_time}")

        self.dataset_length = len(self.samples)    
    
    def load_pickle(self, pkl_path):
        with open(pkl_path, 'rb') as handler:
            _dict = pickle.load(handler)

        return _dict

    def expand_smpl(self, _dict,style,frame_name=None):

        pose = _dict['pose']

        global_orient = torch.Tensor(pose[:3]).unsqueeze(0)
        body_pose     = torch.Tensor(pose[3:72]).unsqueeze(0)


        smpl_output = self.smpl(global_orient=torch.Tensor([[0.0,0.0,0.0]]),body_pose=body_pose,betas=self.betas)

        
        joints_homogen = F.pad(smpl_output.joints[0,0,:], [0, 1])
        
        transform = transform_mat(batch_rodrigues(global_orient),smpl_output.joints[0,0,:].view(-1,3,1))

        rel_transforms = transform - torch.transpose(F.pad(torch.matmul(transform, joints_homogen), [0, 0, 3, 0]),0,1)


        projection_matrix = np.array([[(_dict['f']),0.0,_dict['img_shape'][1]/2.0],[0.0,_dict['f'],_dict['img_shape'][0]/2.0],[0.0,0.0,1.0]])  

        if style == "aist":
            rel_transforms = _dict['global_mat']  
            projection_matrix = np.array([[-1.0*(_dict['f']),0.0,_dict['img_shape'][1]/2.0],[0.0,_dict['f'],_dict['img_shape'][0]/2.0],[0.0,0.0,1.0]])  


        joints_homogen_now = F.pad(smpl_output.joints[0,:,:], [0,1,0, 0],value=1.0)

        correct_joints = torch.matmul(torch.Tensor(rel_transforms), joints_homogen_now.view(-1,4,1)).view(-1,4)[:,:3].cpu().detach().numpy()

        imagePoints,_ = cv2.projectPoints(correct_joints.astype(np.float),np.zeros(3),np.array(_dict['trans']),projection_matrix,np.zeros(5))

        #frame = cv2.imread(frame_name)
        #for i in range(imagePoints.shape[0]):
        #    frame = cv2.circle(frame, (int(imagePoints[i,0,0]),int(imagePoints[i,0,1])), radius=5, color=(0, 0, 255), thickness=-1)

        my_face = [15,24,25,26,27,28,48,53]
        
        face_x = imagePoints[my_face,0,0].mean()
        face_y = imagePoints[my_face,0,1].mean()

        #for i in [15,24,25,26,27,28,48,53]:
        #    frame = cv2.circle(frame, (int(imagePoints[i,0,0]),int(imagePoints[i,0,1])), radius=5, color=(0, 0, 255), thickness=-1)

        max_values = np.zeros(2)
        max_values[0] = np.max(np.abs(imagePoints[:,0,0] - imagePoints[0,0,0]))
        max_values[1] = np.max(np.abs(imagePoints[:,0,1] - imagePoints[0,0,1]))

        min_dim = _dict['img_shape'][0] if (_dict['img_shape'][0] < _dict['img_shape'][1]) else _dict['img_shape'][1]
        min_dim = min_dim/2
                
        box_size = int(np.max(max_values)*1.1)

        box_size = box_size if box_size < min_dim else min_dim

        scale_factor = float(min_dim)/box_size
      
        
        trans_center = [-1.0*(imagePoints[0,0,0] - _dict['img_shape'][1]/2.0),-1.0*(imagePoints[0,0,1] - _dict['img_shape'][0]/2.0)]

        face_x = face_x + trans_center[0]
        face_y = face_y + trans_center[1]

        face_posi = [int(((face_x - _dict['img_shape'][1]/2.0)*scale_factor + min_dim)*((self.render_size_hard/2)/min_dim)),int(((face_y - _dict['img_shape'][0]/2.0)*scale_factor + min_dim)*((self.render_size_hard/2)/min_dim))] 

        #self.render_size_hard  12
        
        
        new_trans = np.reshape(np.array(_dict['trans']),(3))
        
         
        #new_trans = [trans_center[0],trans_center[1],_dict['trans'][2]]
        if style == "aist": 
            new_trans[0] = new_trans[0] - trans_center[0]*new_trans[2]/_dict['f']
        else:
            new_trans[0] = new_trans[0] + trans_center[0]*new_trans[2]/_dict['f']

        new_trans[1] = new_trans[1] + trans_center[1]*new_trans[2]/_dict['f']


        #frame = Image.open(frame_name).convert("RGB")
        
        #frame = affine(frame, 0.0,trans_center, 1.0, [0.0,0.0])

        #import pdb
        #pdb.set_trace()  
        
        rotation_matrix = cv2.Rodrigues(pose[:3])[0]    
  
        rot_vector = np.matmul(rotation_matrix,np.array([0.0,0.0,1.0]))
        _,angle_y = cart2pol(rot_vector[2], rot_vector[0])

        angle_y = angle_y - int((angle_y/(2*np.pi)))*(2*np.pi)
        if angle_y < 0:
            angle_y = angle_y + 2*np.pi

        rot_class = int(angle_y/(2*np.pi/self.num_rot_classes)) 

        if (rot_class > (self.num_rot_classes-1)):
            rot_class = self.num_rot_classes-1
      
        return torch.Tensor(smpl_output.vertices).squeeze(0)[self.in_verts,:], rel_transforms,rot_class,trans_center, new_trans,scale_factor,face_posi

    def __len__(self):
        return(self.dataset_length)
    
    def __getitem__(self, idx):
        sample_dict = self.samples[idx]
        vertices    = sample_dict['vertices']
        trans       = sample_dict['trans']
        #trans_new       = sample_dict['trans_new']
        global_mat  = sample_dict['global_mat']
        f           = sample_dict['f']
        face_posi = sample_dict['face_posi']
        if(self._phase == "train"):
            
            segmentation  = Image.open(sample_dict['seg_file']).convert("L")
            frame         = Image.open(sample_dict['frame']).convert("RGB")
            # make crop: ajust foco, translate, scale
            f = f*sample_dict['scale_factor']
            frame   =  affine(frame, 0.0,sample_dict['trans_center'], 1.0, [0.0,0.0])
            frame   =  affine(frame, 0.0,[0.0,0.0], sample_dict['scale_factor'], [0.0,0.0])            
            segmentation   =  affine(segmentation, 0.0,sample_dict['trans_center'], 1.0, [0.0,0.0])
            segmentation   =  affine(segmentation, 0.0,[0.0,0.0], sample_dict['scale_factor'], [0.0,0.0])

            if self.style == 'mt':
                segmentation_soft = self.transform_soft(segmentation) > 0.9
                segmentation_hard = self.transform_hard(segmentation) > 0.9
            else:
                segmentation_soft = self.transform_soft(segmentation) > 0.0
                segmentation_hard = self.transform_hard(segmentation) > 0.0

            return vertices, segmentation_soft.float(),segmentation_hard.float(),self.transform_soft(frame),self.transform_hard(frame), trans, global_mat,f,face_posi
        return vertices, trans, global_mat,f

def get_dataloaders(args, phase = None, movements=None, test=False):
    steps = [
        ("train", args.dataset_path), 
        ("test", args.test_path)
    ]
    if(phase != None):
        steps = list(filter(lambda x: x[0] == phase, steps))

    datasets = {}

    for phase, data_path in steps:
        if phase == "train":
            trainDataset = RetargetingDataset(
                               data_path,
                               args.source,
                               args.person,
                               args.gender,
                               args.render_size_soft,
                               args.render_size_hard,
                               phase,
                               None,
                               args.style
                           )

            if args.style == "mt":
                samples_weight = torch.from_numpy(trainDataset.samples_weight)
                sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))

                datasets[phase] = torch.utils.data.DataLoader(trainDataset,
                                   batch_size=args.batch_size, 
                                   sampler = sampler, 
                                   num_workers=args.workers, pin_memory=True
                              )
            else:
                datasets[phase] = torch.utils.data.DataLoader(trainDataset,
                                   batch_size=args.batch_size, 
                                   shuffle=True, 
                                   num_workers=args.workers, pin_memory=True
                              )

        else:
            test_datasets = []
            for movement in tqdm(movements, desc="Loading test movements..."):
                testDataset = RetargetingDataset(
                                data_path,
                                args.source,
                                args.test_person,
                                args.gender,
                                args.render_size_soft,
                                args.render_size_hard,
                                phase,
                                movement,
                                args.style
                            )
                test_datasets.append(testDataset)

            testDataset = torch.utils.data.ConcatDataset(test_datasets)

            datasets[phase] = torch.utils.data.DataLoader(testDataset,
                                   batch_size=args.batch_size, 
                                   shuffle=False, 
                                   num_workers=args.workers
                              )
    if test:
        return datasets, test_datasets[0]
    return datasets

def main():

    #main only to test dataset class construction
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", default="S1") ## Setting default args to avoid writting them
    parser.add_argument("-g", "--gender", default="female")
    parser.add_argument("-p", "--person", default="P0")
    parser.add_argument("-d", "--dataset_path", default = "/srv/storage/datasets/thiagoluange/dd_dataset")
    parser.add_argument("-t", "--test_path", default = "/srv/storage/datasets/thiagoluange/dd_dataset")
    parser.add_argument("-t", "--test_path", default = "/srv/storage/datasets/thiagoluange/dd_dataset")
    parser.add_argument("-st", "--style", default="mt")

    args = parser.parse_args()

    dataloaders = get_dataloaders(args)

    for phase in ["train", "test"]:
        if(phase == "test"):
            for idx, (vertices, faces, render_params) in enumerate(dataloaders[phase]):
                print(idx)
                print("Current phase -- {} - Index -- {}".format(phase, idx))
                if(idx > 50):
                    break
        else:
            for idx, (vertices, faces, seg, img, render_params) in enumerate(dataloaders[phase]):
                print(idx)
                print("Current phase -- {} - Index -- {}".format(phase, idx))
                if(idx > 50):
                    break

if __name__ == "__main__":
    main()
