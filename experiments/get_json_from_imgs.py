import json
import glob
import os

# get all from images

from_img_paths = glob.glob('/home/nianyli/Desktop/code/DiffViewTrans/experiments/v1/dataset_varied_yaw_test/*from*')
to_rgb_img_paths = glob.glob('/home/nianyli/Desktop/code/DiffViewTrans/experiments/v1/dataset_varied_yaw_test/*to_rgb*')
to_depth_img_paths = glob.glob('/home/nianyli/Desktop/code/DiffViewTrans/experiments/v1/dataset_varied_yaw_test/*to_depth*')

from_img_paths = [x.split('/')[-1] for x in from_img_paths]
to_rgb_img_paths = [x.split('/')[-1] for x in to_rgb_img_paths]
to_depth_img_paths = [x.split('/')[-1] for x in to_depth_img_paths]

from_img_paths.sort()
to_rgb_img_paths.sort()
to_depth_img_paths.sort()

from_dict = dict()
for name in from_img_paths:
    num = int(name.replace('.png', '').split('_')[-1])
    from_dict[num] = name

# make sure every to rgb and depth has a match
for idx in range(len(to_rgb_img_paths)):
    num_rgb = int(to_rgb_img_paths[idx].replace('.png', '').split('_')[-1])
    num_depth = int(to_depth_img_paths[idx].replace('.png', '').split('_')[-1])
    assert num_rgb == num_depth, "Error: You fucked up.."

to_rgb_dict = dict()
for idx in range(len(to_rgb_img_paths)):
    
    

# what = 'yes'

# from_img