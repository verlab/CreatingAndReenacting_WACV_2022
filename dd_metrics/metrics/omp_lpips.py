import os 
import numpy as np
import sys
from skimage import measure 
import cv2
import tqdm
import compute_dists
from termcolor import colored
import pymp
from models import dist_model as dm

if '-h' in sys.argv or '--help' in sys.argv:
	print 'Generate 3 files with the value of SSMI,LPIPS and MSE metrics. Calculate mean and standard deviation'
	print 'Usage python poisson.py <absolute path to images> <commom name between images> <commom name to generated files> <size of the sliding window>'
	print 'MUST HAVE https://github.com/richzhang/PerceptualSimilarity'
	print 'python comute_dists.py MUST WORK - call lpips.py at root folder of PerceptualSimilarity repository'
else:	
	blend_files = []

	files =  os.listdir(sys.argv[1])
	for file in files:
		if  sys.argv[2] in file:
			blend_files.append(file) 

	blend_files = sorted(blend_files)
	ssmi_dic = pymp.shared.array((len(blend_files)-1,),dtype='float64')
	mse_dic = pymp.shared.array((len(blend_files)-1,),dtype='float64')
	lpips_dic = pymp.shared.array((len(blend_files)-1,),dtype='float64')
	ssmi_file = open(sys.argv[3]+'SSMI.txt','w')
	mse_file = open(sys.argv[3]+'MSE.txt','w')
	lpips_file = open(sys.argv[3]+'LPIPS.txt','w')

	images = []
	print len(blend_files)
	print colored('Reading images...','green')
	for i in tqdm.trange(len(blend_files)-1):
		images.append(cv2.imread(sys.argv[1]+blend_files[i]))


	print colored('Computing metrics...','blue')

	with pymp.Parallel(4) as p: 
		model = dm.DistModel()
		model.initialize(model='net-lin',net='alex',use_gpu=True)
		for idx in range(len(blend_files)-1):
			# i = -int(sys.argv[4])/2
	
		# imageA = cv2.imread(sys.argv[1]+blend_files[idx])
			imageA = images[idx]
			mse_dic[idx] = float('inf')
			ssmi_dic[idx] = 0
			lpips_dic[idx] = float('inf')
			# while i <= int(sys.argv[4])/2:
			for i in range(-int(sys.argv[4])/2,int(sys.argv[4])/2):	
				if (idx+i) < 0 or (i is 0) or (idx+i) > len(images)-1:
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
	file = open(sys.argv[3]+'LPIPS.txt','r')
	lines = file.readlines()
	file.close()
	for line in lines:
		dists.append(float(line.split()[0]))
	# for line in lines:
	#	 if 'model' in line:
	#	 pass
	#	 else:
	#	 dists.append(float(line.split()[0]))

	dists_array = np.asarray(dists)
	file = open(sys.argv[3]+'LPIPS.txt','a')
	file.write('media: '+str(np.mean(dists_array))+'\n')
	file.write('desvio padrao: '+str(np.std(dists_array))+'\n')
	file.close()

	dists = []
	file = open(sys.argv[3]+'SSMI.txt','r')
	lines = file.readlines()
	file.close()
	for line in lines:
		dists.append(float(line.split()[0]))
	dists_array = np.asarray(dists)
	file = open(sys.argv[3]+'SSMI.txt','a')
	file.write('media: '+str(np.mean(dists_array))+'\n')
	file.write('desvio padrao: '+str(np.std(dists_array))+'\n')
	file.close()

	dists = []
	file = open(sys.argv[3]+'MSE.txt','r')
	lines = file.readlines()
	file.close()
	for line in lines:
		dists.append(float(line.split()[0]))
	dists_array = np.asarray(dists)
	file = open(sys.argv[3]+'MSE.txt','a')
	file.write('media: '+str(np.mean(dists_array))+'\n')
	file.write('desvio padrao: '+str(np.std(dists_array))+'\n')
	file.close()
