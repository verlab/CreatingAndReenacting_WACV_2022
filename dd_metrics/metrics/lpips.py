import os 
import numpy as np
import sys
from skimage import measure 
import cv2
import tqdm
import compute_dists
from termcolor import colored
from models import dist_model as dm
import argparse
# import pymp
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="absolute path to input images")
ap.add_argument("-f", "--input_format", required=True,
	help="commom name between images <image format>")
ap.add_argument("-o", "--output", required=True,
	help="commom name to generated files")
ap.add_argument("-n", "--window", required=True,
	help="size of sliding window in frames")
ap.add_argument("-c", "--compare_images", required=True,
	help="absolute path to images which will be compared")
ap.add_argument("-fc", "--compare_format", required=True,
	help="commom name between images <image format>")

_args = ap.parse_args()
_args = vars(_args)


# if '-h' in sys.argv or '--help' in sys.argv:
# 	print 'Generate 3 files with the value of SSMI,LPIPS and MSE metrics. Calculate mean and standard deviation'
# 	print 'Usage python poisson.py <absolute path to images> <commom name between images> <commom name to generated files> <size of the sliding window> <absolute path to images> <commom name between images>'
# 	print 'MUST HAVE https://github.com/richzhang/PerceptualSimilarity'
# 	print 'python comute_dists.py MUST WORK - call lpips.py at root folder of PerceptualSimilarity repository'
# else:	
blend_files = []
compare_files = []

window_size = abs(2*int(_args['window']))
if window_size < 15:
	window_size = 15

files =  os.listdir(_args['input'])
for file in files:
	if  _args['input_format'] in file:
		blend_files.append(file) 
files1 = os.listdir(_args['compare_images'])
for file in files1:
	if _args['compare_format'] in file:
		compare_files.append(file)
compare_files = sorted(compare_files)
blend_files = sorted(blend_files)
ssmi_dic = np.zeros(len(blend_files)-1)
mse_dic = np.full(len(blend_files)-1,np.inf)
lpips_dic = np.full(len(blend_files)-1,np.inf)
ssmi_file = open(_args['output']+'SSMI.txt','w')
mse_file = open(_args['output']+'MSE.txt','w')
lpips_file = open(_args['output']+'LPIPS.txt','w')

images = []
print colored('Reading images...','green')
for i in tqdm.trange(len(compare_files)-1):
	images.append(cv2.imread(_args['compare_images']+compare_files[i]))


model = dm.DistModel()
model.initialize(model='net-lin',net='alex',use_gpu=True)
print colored('Computing metrics...','blue')
for idx in tqdm.trange(len(blend_files)-1):
		# i = -int(sys.argv[4])/2

	imageA = cv2.imread(_args['input']+blend_files[idx])
        imageA = cv2.resize(imageA, (1000, 1080))
	# imageA = images[idx]
	mse_dic[idx] = float('inf')
	ssmi_dic[idx] = 0
	lpips_dic[idx] = float('inf')
		# while i <= int(sys.argv[4])/2:
        
	for i in range(-window_size/2,window_size/2):	
		# if (idx+i) < 0 or (i is 0) or (idx+i) > len(images)-1:
		if (idx+i) < 0 or (idx+i) > len(images)-1:
			pass
		else:
			# imageB = cv2.imread(sys.argv[1]+blend_files[idx+i])
			imageB = images[idx+i]
			##CHANGE LAST PARAMETER TO USE GPU
			l = compute_dists.lpips(imageA[:,:,::-1],imageB[:,:,::-1],True,model)
			m = measure.compare_mse(imageA, imageB)
			s = measure.compare_ssim(imageA, imageB,multichannel=True)

			if m < mse_dic[idx]:
				mse_dic[idx] = m 
			if s > ssmi_dic[idx]:
				ssmi_dic[idx] = s
			if l < lpips_dic[idx]:
				lpips_dic[idx] = l 

			# if m < mse_dic[sys.argv[1]+blend_files[idx]]:
			# 	mse_dic[sys.argv[1]+blend_files[idx]] = m 
			# if s > ssmi_dic[sys.argv[1]+blend_files[idx]]:
			# 	ssmi_dic[sys.argv[1]+blend_files[idx]] = s
			# if l < lpips_dic[sys.argv[1]+blend_files[idx]]:
			# 	lpips_dic[sys.argv[1]+blend_files[idx]] = l 
			# i = i+1
	
print colored('Finish computing metrics.\nWriting files...','blue')
for i in tqdm.trange(len(blend_files)-1):			
	lpips_file.write(str(lpips_dic[i])+'\n')
	ssmi_file.write(str(ssmi_dic[i])+'\n')
	mse_file.write(str(mse_dic[i])+'\n')


####################
ssmi_file.close()
mse_file.close()
lpips_file.close()

dists = []
file = open(_args['output']+'LPIPS.txt','r')
lines = file.readlines()
file.close()
for line in lines:
	dists.append(float(line.split()[0]))
# for line in lines:
#	 if 'model' in line:
	 # pass
#	 else:
#	 dists.append(float(line.split()[0]))

dists_array = np.asarray(dists)
file = open(_args['output']+'LPIPS.txt','a')
file.write('media: '+str(np.mean(dists_array))+'\n')
file.write('desvio padrao: '+str(np.std(dists_array))+'\n')
file.close()

dists = []
file = open(_args['output']+'SSMI.txt','r')
lines = file.readlines()
file.close()
for line in lines:
	dists.append(float(line.split()[0]))
dists_array = np.asarray(dists)
file = open(_args['output']+'SSMI.txt','a')
file.write('media: '+str(np.mean(dists_array))+'\n')
file.write('desvio padrao: '+str(np.std(dists_array))+'\n')
file.close()

dists = []
file = open(_args['output']+'MSE.txt','r')
lines = file.readlines()
file.close()
for line in lines:
	dists.append(float(line.split()[0]))
dists_array = np.asarray(dists)
file = open(_args['output']+'MSE.txt','a')
file.write('media: '+str(np.mean(dists_array))+'\n')
file.write('desvio padrao: '+str(np.std(dists_array))+'\n')
file.close()
