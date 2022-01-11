import numpy as np

def write_obj(out_name,f,v):
    with open(out_name, 'w') as file:
        file.write("# OBJ file\n")
        for vi in range(v.shape[0]):
            file.write("v " + str(v[vi][0]) + " " + str(v[vi][1]) + " " + str(v[vi][2]) + "\n" )
        for fi in range(f.shape[0]):
            file.write("f " + str(f[fi][0] + 1) + " " + str(f[fi][1] + 1) + " " + str(f[fi][2] +1) + "\n" )
