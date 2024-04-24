from utils.ddvt import DDVT
import json
import argparse
from  omegaconf import OmegaConf
import os
import cv2
import numpy as np

def get_imgs_from_different_locations():
    parser = argparse.ArgumentParser()

    parser.add_argument(
            "--verify_dataset", 
            action='store_true', 
            help='Run dataset verification')
        
    parser.add_argument(
        '--train_dataset_path',
        action='store',
        type=str,
        help='Specify the location of the train dataset folder')
    
    parser.add_argument(
        '--trans_label_type',
        action='store',
        type=str,
        choices=['to', 'from'],
        help='Specify the type of translation label')
    
    parser.add_argument(
        '--vid_duration',
        action='store',
        default=60,
        type=int,
        help='Specify the duration of the video to take')
    
    parser.add_argument(
        '--fps',
        action='store',
        default=30,
        type=int,
        help='Specify the fps at which to record a video')
    
    parser.add_argument(
        '--save_dir',
        action='store',
        default='test_save',
        type=str,
        help='Path to store video information.')
    
    parser.add_argument(
        '--world',
        action='store',
        default='town01',
        type=str,
        help='Specify the world to collect data on')
    
    parser.add_argument(
        '--show',
        action='store_true',
        help='Specify whether you want to show the images while running the demo')
    
    parser.add_argument(
        '--spawn_point',
        action='store',
        default='-1',
        type=int,
        help='Specify spawn point. -1 means random')
    
    # default diffusion inference parameters
    parser.add_argument(
        "--diff_ckpt",
        type=str,
        nargs="?",
        help="diffusion model checkpoint path",
    )

    parser.add_argument(
        "--diff_config",
        type=str,
        nargs="?",
        help="diffusion model config path",
    )

    parser.add_argument(
        "--img_save_path",
        type=str,
        help="Specify the path to the directory to save the navigation images to."
    )
    # arguments = parser.parse_args()
    
    # extra argument parsing and processing for DDVT
    opt, unknown = parser.parse_known_args()
    ckpt = opt.diff_ckpt
    opt.base = [opt.diff_config]

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    DDVT_config = OmegaConf.merge(*configs, cli)

    gpu = True
    eval_mode = True
    
    # load the sensor params from the dataset file
    labels_json_path = os.path.join(opt.train_dataset_path,
                                    'labels.json')
    with open(labels_json_path, 'r') as file:
        sensor_params = json.load(file)['sensor_params']
        
    latent_shape = (3, 72, 96)
    
    DDVT_parameters = {
        'config': DDVT_config,
        'ckpt': ckpt,
        'sensor_params': sensor_params,
        'latent_shape': latent_shape,
        'gpu': gpu,
        'eval_mode': eval_mode
    }

    # create model class
    ddvt = DDVT(DDVT_parameters['config'],
                DDVT_parameters['ckpt'],
                DDVT_parameters['sensor_params'],
                DDVT_parameters['latent_shape'],
                DDVT_parameters['eval_mode'],
                DDVT_parameters['gpu'])
    
    img_nums = [8]

    for img_num in img_nums:
        # load the image
        # rgb_img = cv2.imread(f'/home/nianyli/Desktop/code/DDVT/depth_img_{img_num}.png')
        # depth_img = cv2.imread(f'/home/nianyli/Desktop/code/DDVT/depth_img_{img_num}.png')
        rgb_img = cv2.imread('/home/nianyli/Desktop/code/DDVT/scripts/rgb_test.png')
        depth_img = cv2.imread('/home/nianyli/Desktop/code/DDVT/scripts/depth_test.png')


        '''generate an array of translation labels and sweep from:
        left to right, top to bottom and save x, y, z vales of the translation
        '''
        x_values = np.linspace(sensor_params['translation_limits_cube']['x'][0], sensor_params['translation_limits_cube']['x'][1], 10)
        y_values = np.linspace(sensor_params['translation_limits_cube']['y'][0], sensor_params['translation_limits_cube']['y'][1], 10)
        z_values = np.linspace(sensor_params['translation_limits_cube']['z'][0], sensor_params['translation_limits_cube']['z'][1], 5)

        # x_values = x_values[5:-5]
        # y_values = y_values[5:-5]
        # z_values = z_values[5:-5]

        for x_val in x_values:
            for y_val in y_values:
                for z_val in z_values:

                    translation_label = {
                        'x': x_val,
                        'y': y_val,
                        'z': z_val,
                        'yaw': 0
                    }

                    # translation_label = {
                    #     'x': 3.8665317160734873,
                    #     'y': 0.8125471906209163,
                    #     'z': -2.4966923039183295,
                    #     'yaw': 0
                    # }

                    translated_img = ddvt.translate(rgb_img,
                                                    depth_img,
                                                    translation_label,
                                                    preprocessed=False)
                    
                    cv2.imwrite(f"paper_imgs/rgb_translation_{translation_label['x']}_{translation_label['y']}_{translation_label['z']}.png", translated_img[:, :, :3])


    
def use_plotly():
    import plotly.express as px
    df = px.data.iris()
    fig = px.scatter_3d(df, x='sepal_length', y='sepal_width', z='petal_width',
                color='species')
    fig.show()

if __name__=='__main__':
    use_plotly()


