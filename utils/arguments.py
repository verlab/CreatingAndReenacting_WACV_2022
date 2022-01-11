import argparse

def get_args():

    parser = argparse.ArgumentParser()

    ## DATASET PARAMS
    parser.add_argument("-st", "--style", default="mt")

    ## MESH PARAMS
    parser.add_argument("-s", "--source", default="S1")
    parser.add_argument("-g", "--gender", default="female")
    parser.add_argument("-p", "--person", default="P0")
    parser.add_argument("-tp", "--test_person", default="P1")
    parser.add_argument('-ls', '--loss_sil', default=1.0, type=float) # atual: 0.1
    parser.add_argument('-le', '--loss_edge', default=0.0, type=float) 
    parser.add_argument('-lss', '--loss_ssim', default=0.0, type=float) # atual 50.
    parser.add_argument('-ln', '--loss_nor', default=0.5, type=float)
    parser.add_argument('-ll', '--loss_lap', default=1, type=float)
    parser.add_argument('-l', '--lr', default=0.0001, type=float)
    parser.add_argument('-mv', '--movement', default = "bruno")

    ## TEXTURE PARAMS
    parser.add_argument("-wup", "--warm_up", default=2000, type=int) ## Dependendo da pessoa, pode precisar de mais warmup.
    parser.add_argument("-wupg", "--warm_up_gan", default=0, type=int)
    parser.add_argument('-fl', '--flip', default=0.00, type=float)
    parser.add_argument('-lrdf', '--lr_d_factor', default=100.0, type=float)
    parser.add_argument("-gan", "--gan", default = False, action = "store_true", help="Use gan in train.")
    parser.add_argument("-lsgan", "--lsgan", default = True, action = "store_true", help="Use lsgan loss.")
    parser.add_argument('-ltex', '--lr_tex', default=0.002, type=float)
    parser.add_argument('-lres', '--lr_res', default=0.0002, type=float)
    parser.add_argument('-lrgb', '--loss_rgb', default=100, type=float)
    parser.add_argument('-lb', '--lamb', default=1, type=float)
    parser.add_argument("-pm", "--pretrained_path_model", default = '/srv/storage/datasets/thiagoluange/dd_dataset/checkpoints_meshNet/S1P0/meshNet_lr_0.0001-lss_0.0-ls_1.0-ll_1.0-ln_0.5/batch_1/epochs_20/29-10-2020_0:2:17/_model.pth')
    parser.add_argument("-pt", "--pretrained_path_model_tex", default = None)
    parser.add_argument("-pr", "--pretrained_path_model_tex_res", default = None)
    parser.add_argument("-dk", "--dilate_kernel", type=int, default = 3)
    parser.add_argument("-ek", "--erode_kernel", type=int, default = 3)
    


    ## SHARED PARAMS
    parser.add_argument('-e', '--epochs', default=20, type=int) ## Default for texture: 70, talvez precisa de treinar por mais tempo, 100, 200 epochs.
    parser.add_argument('-w', '--workers', default=6, type=int)
    parser.add_argument('-b', '--batch_size', default=8, type=int) ## Default for texture 8.
    parser.add_argument('-test', '--test',default = False, action = "store_true")
    parser.add_argument('-dt', '--delta_test', default=100, type=int)
    parser.add_argument('-sd', '--save_delta', default=10, type=int)
    parser.add_argument('-en', '--experiment_name', default='', type=str)
    parser.add_argument("-f", "--flag", default = None)
    parser.add_argument("-sl", "--save_best_loss", default = False, action = "store_true")
    parser.add_argument("-gpu", "--device", default="0")
    parser.add_argument("-d", "--dataset_path", default = "/srv/storage/datasets/thiagoluange/dd_dataset")
    parser.add_argument("-t", "--test_path", default = "/srv/storage/datasets/thiagoluange/mt-dataset/")
    parser.add_argument("-m", '--model_texture', default = None)
    parser.add_argument('-o', '--output_path', default = None)
    parser.add_argument('-rss', '--render_size_soft', default=256, type=int)
    parser.add_argument('-rsh', '--render_size_hard', default=512, type=int)
    parser.add_argument('-np', '--n_plots', default=10, type=int)

    ## SAVE PARAMS
    
    parser.add_argument("-save", "--save_mesh_texture", default = False, action = "store_true")
     
    
    return parser.parse_args()

def main():
    args = get_args()

if __name__ == "__main__":
    main()
