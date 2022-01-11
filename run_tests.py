import os
import sys
import argparse
import subprocess
from utils.transform import transform_images

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", default="S3")
    parser.add_argument("-pm", "--pretrained_path_model")
    parser.add_argument("-pt", "--pretrained_path_model_tex")
    parser.add_argument("-mv", "--movements", default = "box,cone,simple_walk,hand,fusion,rotate,jump,shake_hands")
    parser.add_argument("-o", "--output_path", default = "/srv/storage/datasets/thiagoluange/dd_dataset/test_results/")
    parser.add_argument("-mtr", "--mt_root", default = "/srv/storage/datasets/thiagoluange/mt-dataset/")
    args = parser.parse_args()

    source = args.source
    mesh_model_fp = args.pretrained_path_model
    tex_model_fp = args.pretrained_path_model_tex 
    movements = args.movements.split(",")
    output_fp = args.output_path
    mt_root = args.mt_root

    for movement in movements:
        args = [
            "python", 
            "test.py", 
            "-s", source, 
            "-pm", mesh_model_fp, 
            "-pt", tex_model_fp, 
            "-mv", movement, 
            "-o", output_fp
        ]

        process = subprocess.Popen(
            args,
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
            universal_newlines = True
        )
        
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        rgb_output_f = os.path.join(output_fp, source, movement, "P1", "rgb")
        transformed_output_f = rgb_output_f.replace("rgb", "transformed")
        os.makedirs(transformed_output_f, exist_ok = True)
        background_path = os.path.join(mt_root, source, movement, "P1", "background")
        background_nf_path = os.path.join(mt_root, source, movement, "P1", "background_not_fill")
        ori_img_path = os.path.join(mt_root, source, movement, "P1", "test_img")

        transform_images(rgb_output_f, background_path, background_nf_path, ori_img_path, transformed_output_f)




if __name__ == "__main__":
    main()
