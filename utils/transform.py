import os
import sys
import cv2
import numpy as np
import re
import argparse
import logging
logger = logging.getLogger(__name__)

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def feather_blending(image,background,mask,edge=3):
    image16 = np.array(image, dtype=np.int16)
    mask8 = mask
    background16 = np.array(background, dtype=np.int16)
    kernel = np.ones((edge, edge), np.uint8)
    background_mask = 255 - cv2.erode(mask8, kernel, iterations=1)
    fb = cv2.detail_FeatherBlender(sharpness=1./edge)
    # corners = [[0, 0], [1080, 0], [1080, 1920], [0, 1920]]
    corners = fb.prepare((0, 0, image.shape[1], image.shape[0]))
    fb.feed(image16, mask8, corners)
    fb.feed(background16, background_mask, corners)
    output = None
    output_mask = None
    output, output_mask = fb.blend(output, output_mask)
    return output


def read_background_images(source, movement, mt_dataset_root):
    background_root = os.path.join(mt_dataset_root, source, movement, "P1", "background")
    images_f = natural_sort(os.listdir(background_root))
    background_images = [cv2.imread(os.path.join(background_root, image_f), -1) for image_f in images_f]
    return background_images

def run_blending(rgb_list, background_list, mask_list, output_path):

    for idx, (rgb_img, background_img, mask_img) in enumerate(zip(rgb_list, background_list, mask_list)):
    
        height_ori, width_ori = background_img.shape[:2]
        min_dim = height_ori if height_ori < width_ori else width_ori
        rgb_img = cv2.resize(rgb_img, (min_dim, min_dim))
        mask_img = cv2.resize(mask_img, (min_dim, min_dim))
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR)
        _, mask_img = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY_INV)

        height_rgb, width_rgb = rgb_img.shape[:2]

        yoff = round((height_ori - height_rgb)/2)
        xoff = round((width_ori - width_rgb)/2)

        #print(yoff, xoff)
    
        result = background_img.copy()
        img_w = np.ones(background_img.shape)*255

        result[yoff : yoff + height_rgb, xoff : xoff + width_rgb] = rgb_img
        img_w[yoff : yoff + height_rgb, xoff : xoff + width_rgb] = mask_img
        
        mask_img = np.array(img_w, dtype=np.uint8)
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)

        _, mask_img = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)
        mask_img = mask_img/float(255)

        #image_seg = (background_img*mask2[:,:,np.newaxis] +  person_seg*(np.ones((person_seg.shape[0],person_seg.shape[1],3)) - mask2[:,:,np.newaxis]))
        mask_mod = (np.ones_like(mask_img.astype(np.uint8)) - mask_img.astype(np.uint8))*255

        feather_seg = feather_blending(result, background_img, mask_mod)
        output_path_img = os.path.join(output_path, "TEST{:05d}.jpg".format(idx))
        cv2.imwrite(output_path_img, feather_seg)


def transform_images(rgb_path, background_path, masks_path, output_path):

    rgb_f = natural_sort(os.listdir(rgb_path))
    background_f = natural_sort(os.listdir(background_path))
    mask_f = natural_sort(os.listdir(masks_path))

    rgb_list = []
    background_list = []
    mask_list = []

    for idx, (rgb_n, background_n, mask_n) in enumerate(zip(rgb_f, background_f, mask_f)):

        rgb_img_fp = os.path.join(rgb_path, rgb_n)
        rgb_img = cv2.imread(rgb_img_fp)

        background_fp = os.path.join(background_path, background_n)
        background_img = cv2.imread(background_fp, -1)

        mask_fp = os.path.join(masks_path, mask_n)
        mask_img = cv2.imread(mask_fp, -1)

        rgb_list.append(rgb_img)
        background_list.append(background_img)
        mask_list.append(mask_img)

    run_blending(rgb_list, background_list, mask_list, output_path)
    os.system('ffmpeg -hide_banner -loglevel panic -framerate 30 -i {}/TEST%05d.jpg {}/result.mp4'.format(output_path, output_path))


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--rgb")
    parser.add_argument("-b", "--background")
    parser.add_argument("-m", "--mask_img")
    parser.add_argument("-o", "--output_path_root", default = "/srv/storage/datasets/thiagoluange/dd_test_results/")
    parser.add_argument("-mt", "--mt_dataset_root", default = "/srv/storage/datasets/thiagoluange/mt-dataset/")
    parser.add_argument("-d", "--data_root", default = None)
    args = parser.parse_args()

    if args.data_root is None:
        rgb_path = args.rgb
        background_path = args.background
        mask_img_path = args.mask_img
        output_path = args.output_path_root
        if os.path.isdir(rgb_path) and os.path.isdir(background_path) and os.path.isdir(mask_img_path):
            os.makedirs(output_path, exist_ok = True)
            logger.info("Blending images from {} to {}".format(rgb_path, output_path))
            transform_images(rgb_path, background_path, mask_img_path, output_path)
    else:
        #works only for mt/dd dataset structure
        data_root = args.data_root
        mt_dataset_root = args.mt_dataset_root
        output_path_root = args.output_path_root

        sources = ["S1", "S2", "S3", "S4"]
        movements = ['box', 'cone', 'fusion', 'hand', 'jump', 'rotate', 'shake_hands', 'simple_walk']

        for source in sources:
            for movement in movements:
                rgb_path = os.path.join(data_root, source, movement, "P1", "rgb")
                background_path = os.path.join(mt_dataset_root, source, movement, "P1", "background")
                mask_img_path = os.path.join(data_root, source, movement, "P1", "mask")
                output_path = os.path.join(output_path_root, source, movement, "P1", "result")

                if os.path.isdir(rgb_path) and os.path.isdir(background_path) and os.path.isdir(mask_img_path):
                    os.makedirs(output_path, exist_ok = True)
                    logger.info("Blending images from {} to {}".format(rgb_path, output_path))
                    transform_images(rgb_path, background_path, mask_img_path, output_path)
                else:

                    for _dir in [rgb_path, background_path, mask_img_path]:
                        if not os.path.isdir(_dir):
                            logger.warn("Dir: {} not found, going to next iter".format(_dir))

if __name__ == '__main__':
    main()
