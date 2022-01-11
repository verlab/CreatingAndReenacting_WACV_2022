import numpy as np
import matplotlib.pyplot as plt
import sys

dists = []
dists.append([])
dists.append([])

file = open(sys.argv[1]+'LPIPS.txt','r')
lines = file.readlines()
file.close()
for line in lines:
    if 'model' in line or 'media' in line or 'desvio' in line:
        pass
    else:
        dists[0].append(float(line.split()[0]))

file = open(sys.argv[2]+'LPIPS.txt','r')
lines = file.readlines()
file.close()
for line in lines:
    if 'model' in line or 'media' in line or 'desvio' in line:
        pass
    else:
        dists[1].append(float(line.split()[0]))

if 'model' in sys.argv[1]:
    model = dists[0]
    unet = dists[1]
else:
    model = dists[1]
    unet = dists[0]

plt.plot(model, label='LPIPS-Model',  marker='', color='blue')
plt.plot(unet, label='LPIPS-vunet',  marker='', color='red')
plt.legend()
plt.show()

dists = []
dists.append([])
dists.append([])

file = open(sys.argv[1]+'SSMI.txt','r')
lines = file.readlines()
file.close()
for line in lines:
    if 'model' in line or 'media' in line or 'desvio' in line:
        pass
    else:
        dists[0].append(float(line))

file = open(sys.argv[2]+'SSMI.txt','r')
lines = file.readlines()
file.close()
for line in lines:
    if 'model' in line or 'media' in line or 'desvio' in line:
        pass
    else:
        dists[1].append(float(line))

if 'model' in sys.argv[1]:
    model = dists[0]
    unet = dists[1]
else:
    model = dists[1]
    unet = dists[0]

plt.plot(model, label='SSMI-Model',  marker='', color='blue')
plt.plot(unet, label='SSMI-vunet',  marker='', color='red')
plt.legend()
plt.show()


dists = []
dists.append([])
dists.append([])

file = open(sys.argv[1]+'MSE.txt','r')
lines = file.readlines()
file.close()
for line in lines:
    if 'model' in line or 'media' in line or 'desvio' in line:
        pass
    else:
        dists[0].append(float(line))

file = open(sys.argv[2]+'MSE.txt','r')
lines = file.readlines()
file.close()
for line in lines:
    if 'model' in line or 'media' in line or 'desvio' in line:
        pass
    else:
        dists[1].append(float(line))

if 'model' in sys.argv[1]:
    model = dists[0]
    unet = dists[1]
else:
    model = dists[1]
    unet = dists[0]

plt.plot(model, label='MSE-Model',  marker='', color='blue')
plt.plot(unet, label='MSE-vunet',  marker='', color='red')
plt.legend()
plt.show()
