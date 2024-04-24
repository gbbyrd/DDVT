# import glob
# import os
# import sys
# import time
# import random
# import numpy as np
# import cv2

# import carla

# def main():
#     client = carla.Client('localhost', 2000)
    
#     actor_list = []

#     # Load Town01 map
#     world = client.load_world('Town01')

#     blueprint_library = world.get_blueprint_library()

#     # Spawn a random vehicle at a random location
#     spawn_point = random.choice(world.get_map().get_spawn_points())
#     vehicle = world.spawn_actor(blueprint_library.filter('model3')[0], spawn_point)
#     actor_list.append(vehicle)

#     # Create an RGB sensor with fov 100 and image size 256 by 256
#     camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
#     camera_bp.set_attribute('fov', '100')
#     camera_bp.set_attribute('image_size_x', '256')
#     camera_bp.set_attribute('image_size_y', '256')
#     camera_transform = carla.Transform(carla.Location(x=-3.5, y=0, z=3.5), carla.Rotation(pitch=-20))
#     camera_rgb_1 = world.spawn_actor(camera_bp, camera_transform, vehicle)
#     actor_list.append(camera_rgb_1)

#     # Create an RGB sensor with fov 100 and image size 400 by 300
#     camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
#     camera_bp.set_attribute('fov', '100')
#     camera_bp.set_attribute('image_size_x', '256')
#     camera_bp.set_attribute('image_size_y', '256')
#     camera_transform = carla.Transform(carla.Location(x=-2.5, y=0, z=3.5), carla.Rotation(pitch=-20))
#     camera_rgb_2 = world.spawn_actor(camera_bp, camera_transform, vehicle)
#     actor_list.append(camera_rgb_2)
    
#     def save_carla_sensor_img(image, img_name, base_path, dim):
#         """Processes a carla sensor.Image object and a velocity int

#         Args:
#             image (Carla image object): image object to be post processed
#             img_name (str): name of the image
#             base_path (str): file path to the save folder

#         Returns:
#             i3 (numpy array): numpy array of the rgb data of the image
#         """
#         IM_HEIGHT, IM_WIDTH = dim
        
#         i = np.array(image.raw_data) # the raw data is initially just a 1 x n array
#         i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4)) # raw data is in rgba format
#         i3 = i2[:, :, :3] # we just want the rgb data
#         print(i3.shape)
#         # save the image
#         full_save_path = os.path.join(base_path,
#                                         img_name)
#         cv2.imwrite(full_save_path, i3)
#         return i3

#     # Collect and save one image each to the same directory as the file
#     for _ in range(1):
#         image_rgb_1 = camera_rgb_1.listen(lambda image: save_carla_sensor_img(image, 'img_1.png', '.', (256, 256)))
#         image_rgb_2 = camera_rgb_2.listen(lambda image: save_carla_sensor_img(image, 'img_2.png', '.', (256, 256)))

#     time.sleep(1.0)

#     for actor in actor_list:
#         actor.destroy()

# def save_image(image, filename):
#     image.save_png(filename)

# if __name__ == '__main__':
#     try:
#         main()
        
#     except:
#         print('oops')
    
# import cv2
# import glob
# import numpy as np

# def preprocess_depth(depth_img_path):
#         """ Normalize depth image and return h x w x 1 numpy array."""
#         depth_img = cv2.imread(depth_img_path)
#         depth_img = depth_img[:,:,2] + 256 * depth_img[:,:,1] + 256 * 256 * depth_img[:,:,0]
#         depth_img = depth_img / (256 * 256 * 256 - 1)
        
#         return depth_img
    
# img_paths = glob.glob('/home/nianyli/Desktop/code/DiffViewTrans/experiments/v1/dataset/*depth*')
# img_paths.sort()
# results = []
# for idx, depth_img_path in enumerate(img_paths):
#     depth_img = preprocess_depth(depth_img_path)
#     results.append({
#         'img': depth_img_path,
#         'min_value': np.min(depth_img)
#     })

# results = sorted(results, key=lambda x: x['min_value'])
# results.reverse()

# rgb_img_paths = glob.glob('/home/nianyli/Desktop/code/DiffViewTrans/experiments/v1/dataset/*rgb*')
# for i in results:
#     # get number
#     depth_img_name = i['img'].split('/')[-1].split('.')[0]
#     num = depth_img_name[-6:]
#     for rgb_img_path in rgb_img_paths:
#         if num in rgb_img_path:
#             rgb_img = cv2.imread(rgb_img_path)
#             print(i['min_value'])
#             cv2.imshow('test', rgb_img)
#             cv2.waitKey(0)
    
import ijson

data = ijson.parse(open('/home/nianyli/Desktop/code/DiffViewTrans/experiments/v1/dataset/labels.json', 'r'))
what = 'yes'
for prefix, event, value in data:
    print(prefix, event, value)
    breakpoint()
    