import carla

import os
import sys
import random
import time
import numpy as np
import cv2
import shutil
import pygame
import queue
import json
import glob
from tqdm import tqdm
import argparse

from math import sqrt

"""
This script was used to collect data with generated traffic. The generate_traffic.py
script had to be run separately and then this script was ran. This script filters
out occlusions using various proximity filters for both the auxiliary sensor and
target sensor.
"""

# global variables for quick data collection finetuning
DATASET_PATH = '/home/nianyli/Desktop/code/thesis/DiffViewTrans/data/vt_town01_dataset'
NUM_FRAMES = 100
IM_HEIGHT, IM_WIDTH = 288, 384

# define the x and z locations that the 'to' spawned cameras will spawn at
FRONT_X = 1.3
FRONT_Z = 2.3

class CarlaSyncMode(object):
    """Class for running Carla in synchronous mode. Allows frame rate sync to
    ensure no data is lost and all backend code runs and is completed before
    'ticking' to the next frame."""
    
    def __init__(self, world, **kwargs):
        self.world = world
        self.actors = []
        self.vehicle = kwargs.get('vehicle', None)
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        self._queues = []
        self._settings = None
        self.sensor_params = kwargs.get('sensor_params', None)
        self.blueprint_library = world.get_blueprint_library()
        
        self.auxiliary_sensors = []
        self.auxiliary_sensor_info = []
        
        self.target_sensors = []
        self.target_sensor_info = []
        
    def __enter__(self):
        # get some basic carla metadata required for functionality
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode = False,
            synchronous_mode = True, 
            fixed_delta_seconds = self.delta_seconds
        ))
        
        # create queues for each sensor that store the data as the sensor reads it
        self.make_queue(self.world.on_tick)

        self.actors.append(self.vehicle)
        
        self.randomize_auxiliary_sensors()
        
        return self
    
    def make_queue(self, register_event):
        # create a q for the event to register data to
        q = queue.Queue()
        # define q.put as the function that is called when the event recieves data
        register_event(q.put)
        # add q to the list of _queues
        self._queues.append(q)
    
    def tick(self, timeout):
        """Call this function to step one frame through the simulation"""
        
        # get the next frame from the world.. this should automatically 
        # update the data in the sensor queues as well
        self.frame = self.world.tick()
        
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data
    
    def remove_sensors_from_queue(self):
        self._queues = self._queues[:1]
     
    def __exit__(self, *args, **kwargs):
        # make sure to clean up the memory
        self.world.apply_settings(self._settings)
        for auxiliary_sensor in self.auxiliary_sensors:
            auxiliary_sensor.destroy()
        for target_sensor in self.target_sensors:
            target_sensor.destroy()
        for actor in self.actors:
            actor.destroy()
        
    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data
            
            

            
    ############################################################################
    ########################## CUSTOM CLASS METHODS ############################
    ############################################################################
    
    def initialize_front_sensors(self):
        """Create and spawn all sensor types at front location"""
        
        num_aux_sensors = self.sensor_params['num_aux_sensors']
        sensor_types = self.sensor_params['sensor_types']
        blueprint_attributes = self.sensor_params['blueprint_attributes']
        initial_spawn_limits = self.sensor_params['initial_spawn_limits']
        relative_spawn_limits = self.sensor_params['relative_spawn_limits']
        
        for sensor_type in sensor_types:
            # create each sensor
            sensor_bp = self.blueprint_library.find(sensor_type)
            for attribute in blueprint_attributes:
                sensor_bp.set_attribute(attribute, blueprint_attributes[attribute])

            # spawn the sensor at 0,0,0
            spawn_point = carla.Transform(carla.Location(x=FRONT_X, y=0, z=FRONT_Z),
                                            carla.Rotation(roll=0, pitch=0, yaw=0))
            sensor = self.world.spawn_actor(sensor_bp, spawn_point, attach_to=self.vehicle)
            self.sensors.append(sensor)

        # # add each sensor to the queue
        # for sensor in self.sensors:
        #     self.make_queue(sensor.listen)
    
    def randomize_auxiliary_sensors(self):
        """Clear the previous auxiliary sensor and spawn a new auxiliary sensor
        in a random location based on the sensor parameters.
        """
        
        sensor_types = self.sensor_params['sensor_types']
        blueprint_attributes = self.sensor_params['blueprint_attributes']
        auxiliary_sensor_spawn_limits = self.sensor_params['auxiliary_sensor_spawn_limits']
        
        # destroy and clean up previous auxiliary sensors and info
        for sensor in self.auxiliary_sensors:
            sensor.destroy()
        self.auxiliary_sensors = []
        self.auxiliary_sensor_info = []
        
        # get random spawn point for auxiliary sensor
        x_lim_aux = auxiliary_sensor_spawn_limits['x']
        y_lim_aux = auxiliary_sensor_spawn_limits['y']
        z_lim_aux = auxiliary_sensor_spawn_limits['z']
        roll_lim_aux = auxiliary_sensor_spawn_limits['roll']
        pitch_lim_aux = auxiliary_sensor_spawn_limits['pitch']
        yaw_lim_aux = auxiliary_sensor_spawn_limits['yaw']
        
        x_aux = random.uniform(x_lim_aux[0], x_lim_aux[1])
        y_aux = random.uniform(y_lim_aux[0], y_lim_aux[1])
        z_aux = random.uniform(z_lim_aux[0], z_lim_aux[1])
        roll_aux = random.uniform(roll_lim_aux[0], roll_lim_aux[1])
        pitch_aux = random.uniform(pitch_lim_aux[0], pitch_lim_aux[1])
        yaw_aux = random.uniform(yaw_lim_aux[0], yaw_lim_aux[1])
        
        # spawn each sensor type at the random location and orientation
        for sensor_type in sensor_types:
            # create sensor
            sensor_bp = self.blueprint_library.find(sensor_type)
            for attribute in blueprint_attributes:
                sensor_bp.set_attribute(attribute, blueprint_attributes[attribute])

            spawn_point = carla.Transform(carla.Location(x=x_aux, y=y_aux, z=z_aux),
                                            carla.Rotation(roll=roll_aux, yaw=yaw_aux, pitch=pitch_aux))
            
            # spawn the sensor relative to the first sensor
            sensor = self.world.spawn_actor(sensor_bp, spawn_point, attach_to=self.vehicle)
            self.auxiliary_sensors.append(sensor)
            
            # save the aux sensor location relative to the ego vehicle
            auxiliary_location_and_orientation = {
                'x': x_aux,
                'y': y_aux,
                'z': z_aux,
                'yaw': yaw_aux,
                'pitch': pitch_aux,
                'roll': roll_aux
            }
            self.auxiliary_sensor_info.append(auxiliary_location_and_orientation)
            
        # clear old sensor queues and add new sensors to queues
        if len(self._queues) > 1:
            self.remove_sensors_from_queue()
        for auxiliary_sensor in self.auxiliary_sensors:
            self.make_queue(auxiliary_sensor.listen)
            
    def randomize_target_sensors(self):
        """ Spawn <num_target_sensors> target sensors in random locations 
        within the translation limits cube of the auxiliary sensor.
        """
        
        num_target_sensors = self.sensor_params['num_target_sensors']
        sensor_types = self.sensor_params['sensor_types']
        blueprint_attributes = self.sensor_params['blueprint_attributes']
        translation_limits_cube = self.sensor_params['translation_limits_cube']
        
        # destroy and clean previous target sensors and info
        for sensor in self.target_sensors:
            sensor.destroy()
        self.target_sensors = []
        self.target_sensor_info = []
        
        # for each sensor in num sensors, get a random location in the spawned
        # auxiliary sensor's translation limits cube and spawn each sensor type
        for idx in range(num_target_sensors):
            
            # get spawn limits for target sensors 
            x_lim_rel = translation_limits_cube['x'].copy()
            y_lim_rel = translation_limits_cube['y'].copy()
            z_lim_rel = translation_limits_cube['z'].copy()
            roll_lim_rel = translation_limits_cube['roll'].copy()
            pitch_lim_rel = translation_limits_cube['pitch'].copy()
            yaw_lim_rel = translation_limits_cube['yaw'].copy()
            
            x_rel = random.uniform(x_lim_rel[0], x_lim_rel[1])
            y_rel = random.uniform(y_lim_rel[0], y_lim_rel[1])
            z_rel = random.uniform(z_lim_rel[0], z_lim_rel[1])
            roll_rel = random.uniform(roll_lim_rel[0], roll_lim_rel[1])
            pitch_rel = random.uniform(pitch_lim_rel[0], pitch_lim_rel[1])
            yaw_rel = random.uniform(yaw_lim_rel[0], yaw_lim_rel[1])
            
            # spawn the new sensors
            for sensor_type in sensor_types:
                # create sensor
                sensor_bp = self.blueprint_library.find(sensor_type)
                for attribute in blueprint_attributes:
                    sensor_bp.set_attribute(attribute, blueprint_attributes[attribute])

                # check to make sure the camera is not being spawned to low to
                # the road
                z_init = self.auxiliary_sensor_info[0]['z']
                z_global = z_init + z_rel
                assert z_global >= 0.5, "Error: z_rel value too low."

                spawn_point = carla.Transform(carla.Location(x=x_rel, y=y_rel, z=z_rel),
                                              carla.Rotation(roll=roll_rel, yaw=yaw_rel, pitch=pitch_rel))
                
                # spawn the target sensor relative to the auxiliary sensor
                target_sensor = self.world.spawn_actor(sensor_bp, spawn_point, attach_to=self.auxiliary_sensors[0])
                self.target_sensors.append(target_sensor)
                
                # save the location of the initial sensor relative to the auxiliary sensor
                relative_location = {
                    'x': x_rel,
                    'y': y_rel,
                    'z': z_rel,
                    'yaw': yaw_rel,
                    'roll': roll_rel,
                    'pitch': pitch_rel
                }
                self.target_sensor_info.append(relative_location)
                
        # ensure that the previous target sensor queues have been deleted and
        # that the queues contain auxiliary sensor queues
        assert len(self._queues) == 1+len(sensor_types), ("Error: Did you forget )" \
            "to clear target sensor queues or spawn a new auxiliary sensor?")
                
        # add the sensors to queues
        for target_sensor in self.target_sensors:
            self.make_queue(target_sensor.listen)

def save_carla_sensor_img(image, img_name, base_path):
    """Processes a carla sensor.Image object and a velocity int

    Args:
        image (Carla image object): image object to be post processed
        img_name (str): name of the image
        base_path (str): file path to the save folder

    Returns:
        i3 (numpy array): numpy array of the rgb data of the image
    """
    i = np.array(image.raw_data) # the raw data is initially just a 1 x n array
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4)) # raw data is in rgba format
    i3 = i2[:, :, :3] # we just want the rgb data
    
    # save the image
    full_save_path = os.path.join(base_path,
                                    img_name)
    cv2.imwrite(full_save_path, i3)
    return i3

def get_img_from_sensor_data(image):
    """Processes a carla sensor.Image object and a velocity int

    Args:
        image (Carla image object): image object to be post processed
        img_name (str): name of the image
        base_path (str): file path to the save folder

    Returns:
        i3 (numpy array): numpy array of the rgb data of the image
    """
    i = np.array(image.raw_data) # the raw data is initially just a 1 x n array
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4)) # raw data is in rgba format
    i3 = i2[:, :, :3] # we just want the rgb data
    
    return i3

def proximity_filter_aux_sensors(depth_img):
    """ The auxiliary sensors should not be spawned where the view is too
    severely occluded. For this, a bit of occlusion is okay, but more occlusion
    should cause this function to filter out the image.
    
    Args:
        depth_image (np array): unnormalized depth image (https://carla.readthedocs.io/en/stable/cameras_and_sensors/#camera-depth-map)
    
    Returns:
        bool: True if the sensor passes, False if filtered out by proximity filter
    """
    
    # define the filter thresholds
    min_acceptable_distance = 0.004 # will be empirically optimized
    
    # the below is the max percentage of the image that can have objects closer
    # than the specified min acceptable distance
    max_acceptable_occlusion = 0.1 
    
    # get normalized depth image (image is in BGR format) between 0 and 1
    normalized = ((depth_img[:, :, 2] 
                  + 256 * depth_img[:, :, 1] 
                  + 256 * 256 * depth_img[:, :, 0])
                  / (256 * 256 * 256 - 1))
    
    # get the number of pixels that are too close
    temp = normalized < min_acceptable_distance
    
    num_too_close = np.sum(temp)
    
    max_close_pixels = IM_HEIGHT * IM_WIDTH * max_acceptable_occlusion
    
    if num_too_close > max_close_pixels:
        return False
    
    return True

def proximity_filter_target_sensors(depth_img):
    """ We do not want to collect target sensor images that are spawned inside 
    other objects. There is not a good way to prevent this in Carla, so this
    function will use the depth image to determine if an object is too close to 
    the camera. If the object is too close to the camera, then the image will be 
    discarded.
    
    Args:
        depth_image (np array): unnormalized depth image (https://carla.readthedocs.io/en/stable/cameras_and_sensors/#camera-depth-map)
    
    Returns:
        bool: True if the sensor passes, False if filtered out by proximity filter
    """
    # define the filter thresholds
    # this value was found emperically. should ensure that the cameras do not
    # spawn inside most vehicles
    min_acceptable_distance = 0.0005 
    
    # get normalized depth image (image is in BGR format) between 0 and 1
    normalized = ((depth_img[:, :, 2] 
                  + 256 * depth_img[:, :, 1] 
                  + 256 * 256 * depth_img[:, :, 0])
                  / (256 * 256 * 256 - 1))
    
    min_depth = np.min(normalized)
    
    if min_depth < min_acceptable_distance:
        print(min_depth)
        return False
    
    return True

def get_img_info_dict(data_type, sensor_type, train_img_count):
    train_img_name = f'{data_type}_{sensor_type}_{str(train_img_count).zfill(6)}.png'
    train_img_info = dict()
    train_img_info['img_name'] = train_img_name
    train_img_info['location'] = {
        'x': 0,
        'y': 0,
        'z': 0,
        'yaw': 0,
        'pitch': 0,
        'roll': 0
    }
    train_img_info['sensor_type'] = sensor_type
    
    return train_img_info

def main():
    
    # make life easier..
    if not os.path.exists(args.dataset_path):
        os.makedirs(args.dataset_path)
        
    actor_list = []

    try:
        # carla boilerplate variables
        client = carla.Client('127.0.0.1', 2000)
        client.set_timeout(2.0)
        if args.world is not None:
            world = client.load_world(args.world)
        world = client.get_world()
        blueprint_library = world.get_blueprint_library()
        clock = pygame.time.Clock()
        
        ########################################################################
        # define sensor parameters (fine tune data control)
        ########################################################################
        
        """ The data collection works in the following way:

        1. Spawn an expert vehicle that runs on Carla's autopilot.
        2. At each frame:
            a. Spawn an auxiliary sensor that represents the 'from' image at a
                random location that is within defined boundaries
            b. Spawn multiple target sensors off of that initial sensor
                randomly, within a set of defined boundaries
        3. Collect the depth, rgb, and semantic segmentation images for the
            dataset
        
        """

        # define initial sensor spawning limits relative to the ego vehicle
        
        # CAREFUL: these limits are from the initial sensor relative to the car 
        auxiliary_sensor_spawn_limits = {
            'x': [-3.5, 3.5],
            'y': [-3, 3],
            'z': [3.5, 5.5],
            'roll': [0, 0],
            'pitch': [0, 0],
            'yaw': [-180, 180]        
        }
        
        # define how far an auxiliary sensor can spawn from the initial sensor
        # note: the aux sensors will always spawn on the x and z axis at the front of
        # the vehicle, but the y value will vary to ensure that the model learns
        # to use the distance label

        # CAREFUL: these limits are from the auxiliary sensor relative to the initial sensor
        translation_limits_cube = {
            'x': [1, 4],
            'y': [-2, 2],
            'z': [-3, 3],
            'roll': [0, 0],
            'pitch': [0, 0],
            'yaw': [0, 0]
        }
        
        blueprint_attributes = {
            'image_size_x': f"{IM_WIDTH}",
            'image_size_y': F"{IM_HEIGHT}",
            'fov': "100"
        }
        
        # define what sensors to collect data from at each spawn point
        num_target_sensors = 10 
        sensor_types = [
            "sensor.camera.depth",                      # * depth should always be first
            # "sensor.camera.semantic_segmentation",
            # "sensor.camera.instance_segmentation",
            "sensor.camera.rgb"
        ]
        
        sensor_params = {
            'num_target_sensors': num_target_sensors,
            'sensor_types': sensor_types,
            'blueprint_attributes': blueprint_attributes,
            'auxiliary_sensor_spawn_limits': auxiliary_sensor_spawn_limits,
            'translation_limits_cube': translation_limits_cube
        }
        
        ########################################################################
        # functionality for adding to existing dataset
        ########################################################################
        
        # search to see if there are any images/labels in the data directory
        img_files = glob.glob(args.dataset_path+'/*.png')
        
        # if files were found, do some additional work to add data to the dataset
        if len(img_files) > 0:
            # load the labels file
            with open(args.dataset_path+'/labels.json', 'r') as file:
                prev_data = json.load(file)
                prev_label_data = prev_data['data']
                prev_sensor_params = prev_data['sensor_params']
        
            # find the count of the final image and set the train_img_count
            train_img_count = len(img_files) // len(sensor_types)
            
            # increase NUM_IMAGES by the initial train_img_count
            num_images = train_img_count + args.num_frames
            
            print(f"initial len: {len(prev_label_data)}")
        else:
            train_img_count = 0
            prev_label_data = None
            prev_sensor_params = None
            num_images = args.num_frames

        ########################################################################
        # create ego vehicle
        ########################################################################
        # get vehicle blueprint
        bp = blueprint_library.filter("model3")[0]
        # bp = random.choice(blueprint_library.filter('model3'))
        
        # spawn vehicle
        spawn_point1 = random.choice(world.get_map().get_spawn_points()) # get a random spawn point from those in the map
        spawn_points = world.get_map().get_spawn_points()  
        idx = random.randint(0, len(spawn_points))
        # spawn_point1 = spawn_points[idx]
        spawn_point1 = spawn_points[0]
        vehicle1 = world.spawn_actor(bp, spawn_point1) # spawn car at the random spawn point
        actor_list.append(vehicle1)
        
        # need to wait for the generate traffic script
        # time.sleep(20)

        ########################################################################
        # start the simulation
        ########################################################################
        
        with CarlaSyncMode(world, fps=30, vehicle=vehicle1, sensor_params=sensor_params) as sync_mode:
            
            # if you are adding to a previous dataset, ensure your dataset parameters
            # are the same
            if prev_sensor_params:
                assert prev_sensor_params == sensor_params, ("Error: To add to a dataset " \
                    "you must have identical sensor parameters.")
            
            # create variables for storing data
            labels = dict()
            sensor_groups = []
            
            pbar = tqdm(desc='Generating training images', total=num_images-train_img_count)
            
            frame_count = 0
            sync_mode.remove_sensors_from_queue()
            sync_mode.randomize_auxiliary_sensors()
            sync_mode.randomize_target_sensors()
            sync_mode.vehicle.set_autopilot(True)
            
            # run simulation and collect data
            while train_img_count < num_images:
                
                print('new frame')
                
                # get frame information
                clock.tick()
                world_data  = sync_mode.tick(timeout=2.0)        
                snapshot = world_data[0]
                auxiliary_sensor_data = world_data[1:1+len(sensor_types)]
                target_sensor_data = world_data[1+len(sensor_types):]
                sim_fps = round(1.0 / snapshot.timestamp.delta_seconds)
                true_fps = clock.get_fps()

                # only collect data every <args.skip_frames> frame
                if frame_count % args.skip_frames != 0:
                    frame_count += 1
                    continue

                # if the car is stopped (like at a stop light) do not collect data
                # as this will heavily bias the dataset
                # TODO: Replace this by calculating the SSIM of the current image
                # and previous image. This will ensure that multiple images are not
                # too similar (which would bias the dataset)
                
                # vel = sqrt(sync_mode.vehicle.get_velocity().x**2 + sync_mode.vehicle.get_velocity().y**2)
                # if vel < .001:
                #     frame_count += 1
                #     continue
                
                # define dictionary for saving the frame data
                sensor_group_data = dict()
                
                # save the rest of the training imgs
                save_labels = True
                imgs_to_save = []
                saved_train_img_count = train_img_count
                
                # get auxiliary sensor data
                for idx, data in enumerate(auxiliary_sensor_data):
                    
                    sensor_type = sensor_types[idx % len(sensor_types)].split('.')[-1]
                    
                    # get image numpy array
                    img = get_img_from_sensor_data(data)
                    
                    # check for occlusions
                    if sensor_type == 'depth':
                        if not proximity_filter_aux_sensors(img):
                            save_labels = False
                            break
                    
                    # add img info dict to sensor group list
                    data_type = 'from'
                    train_img_info = get_img_info_dict(data_type, sensor_type, train_img_count)
                    sensor_group_data[f'{sensor_type}_img_{train_img_count}_info'] = train_img_info
                    
                    # add the img to the list of imgs to save at the end of the
                    # iteration
                    imgs_to_save.append({
                        'img_np_array': img,
                        'img_name': train_img_info['img_name']
                    })
                    
                    sensor_group_data[f'{sensor_type}_img_{train_img_count}_info'] = train_img_info
                    
                if not save_labels:
                    # will not iterate frame count here. since we are skipping
                    # these images, we want the next frame to be recorded
                    print('auxiliary sensor occluded')
                    continue
                    
                # increment train_img_count if auxiliary sensors are valid
                train_img_count += 1
                        
                # get target sensor data
                skip_next_count = 0
                
                # for the target sensors, the save_labels variable will initially be
                # false and will become true if any of the target sensors are valid
                save_labels = False
                for idx, data in enumerate(target_sensor_data):
                    
                    sensor_type = sensor_types[idx % len(sensor_types)].split('.')[-1]
                    
                    img = get_img_from_sensor_data(data)
                    
                    # check for occlusions
                    if sensor_type == 'depth':
                        if not proximity_filter_target_sensors(img):
                            skip_next_count = len(sensor_types)-1
                            continue
                        
                    # skip the next sensors if the corresponding depth sensor
                    # was filtered out by the proximity filter
                    elif skip_next_count > 0:
                        skip_next_count -= 1
                        continue
                    
                    # if even one target sensor is valid, change the save_labels
                    # variable to true
                    save_labels = True
                    
                    # add img info dict to sensor group list
                    data_type = 'to'
                    train_img_info = get_img_info_dict(data_type, sensor_type, train_img_count)
                    sensor_group_data[f'{sensor_type}_img_{train_img_count}_info'] = train_img_info
                    
                    imgs_to_save.append({
                        'img_np_array': img,
                        'img_name': train_img_info['img_name']
                    })
                    
                    # save label
                    sensor_group_data[f'{sensor_type}_img_{train_img_count}_info'] = train_img_info
                    
                    # increment train_img_count every -len(sensor_types)- img saves
                    if (idx+1) % len(sensor_types) == 0:
                        train_img_count += 1
                        
                if save_labels:
                    print('should be saving')
                    # save all of the images in the img_to_save list
                    for img_info in imgs_to_save:
                        img_save_path = os.path.join(args.dataset_path,
                                                     img_info['img_name'])
                        cv2.imwrite(img_save_path, img_info['img_np_array'])
                        
                    sensor_groups.append(sensor_group_data)
                    # increment frame count and the terminal progress par
                    pbar.update(len(target_sensor_data)//len(sensor_types))
                    frame_count += 1
                else:
                    # in this case, the auxiliary sensor was valid, but none of
                    # the target sensors were valid, so we must reset the train_img_count
                    # and will not save any images
                    print('all target sensors occluded')
                    train_img_count = saved_train_img_count
                    
                # randomize sensor locations
                sync_mode.remove_sensors_from_queue()
                sync_mode.randomize_auxiliary_sensors()
                sync_mode.randomize_target_sensors()
                
            labels['data'] = sensor_groups
            if prev_label_data:
                labels['data'] += prev_label_data
            labels['sensor_params'] = sensor_params
            
            labels_path = os.path.join(args.dataset_path,
                                       'labels.json')
            
            os.remove(labels_path)
            print(f"final len: {len(labels['data'])}")
            with open(labels_path, 'w') as file:
                json.dump(labels, file)
                
        # create a synchronous mode context
        time.sleep(1)
        
    finally:
        for actor in actor_list:
            actor.destroy()
        
        # save the labels if there is an error that broke the simulation before
        # completing the data collection
        labels['data'] = sensor_groups
        if prev_label_data:
            labels['data'] += prev_label_data
        labels['sensor_params'] = sensor_params
        
        labels_path = os.path.join(args.dataset_path,
                                    'labels.json')
        with open(labels_path, 'w') as file:
            json.dump(labels, file)
            
        print("All cleaned up!")
        
def verify_dataset():
    """This function is meant to ensure that the above collected dataset adheres
    to the following rules:
    
    1. Every img name in the json labels file has a corresponding saved image in
    the dataset file path.
    2. Every saved image file has a corresponding json labels file img name.
    3. Every collected image from each sensor is sequential with no gaps.
    
    WARNING: This only works if you collect sensor data from CAMERAS
    """

    # TODO: There is an error in the way the data is collected. Switch the 
    # sensor type in the img name from 'semantic' to 'semantic_segmentation'
    dataset_path = args.dataset_path

    # get the dataset information
    with open(dataset_path+'/labels.json', 'r') as file:
        dataset = json.load(file)
        
    sensor_params = dataset['sensor_params']
    data = dataset['data']
    
    sensor_types = sensor_params['sensor_types']
    for idx, sensor_type in enumerate(sensor_types):
        sensor_types[idx] = sensor_type.split('.')[-1]
    
    # get img names in the dataset
    dataset_img_names = set()
    for group_info in data:
        for img in group_info:
            dataset_img_names.add(group_info[img]['img_name'])
    
    # get all of the saved images
    saved_img_paths = glob.glob(dataset_path+'/*.png')
    
    for idx, saved_img_path in enumerate(saved_img_paths):
        saved_img_paths[idx] = saved_img_path.split('/')[-1]
    
    saved_imgs = dict()
    for sensor_type in sensor_types:
        saved_imgs[sensor_type] = []
    
    for saved_img_path in saved_img_paths:
        sensor_type = saved_img_path.split('_')[1]

        assert sensor_type in saved_imgs, 'Error: invalid sensor type in img name!'

        saved_imgs[sensor_type].append(saved_img_path)
    
    def get_substr(string):
        return string[-10:-4]
    
    for sensor_type in saved_imgs:
        # TODO: Fix the below hardcoding.. ew
        saved_imgs[sensor_type].sort(key=lambda x: x[-10:-4])
    
    # verify that every saved image has a corresponding image name in the labels
    # file and that the images are saved in sequential order with no numbers
    # missing
    assert len(dataset_img_names) == len(saved_img_paths), 'Error: Mismatched saved images and labels!'
    
    imgs_not_saved = dict()
    for sensor_type in sensor_types:
        imgs_not_saved[sensor_type] = []

    depth_imgs_not_saved = []
    rgb_imgs_not_saved = []
    semantic_segmentation_imgs_not_saved = []
    
    for sensor_type in saved_imgs:
        count = 0
        idx = 0
        while idx < len(saved_imgs[sensor_type]):
            img_path = saved_imgs[sensor_type][idx]
            if int(img_path[-10:-4]) == count:
                idx += 1
            else:
                imgs_not_saved[sensor_type].append(str(count).zfill(6))

            count += 1
        
    # loop through each saved image and ensure that there is a corresponding label
    # for that image
    missing_labels = dict()
    for sensor_type in sensor_types:
        missing_labels[sensor_type] = []

    missing_depth_labels = []
    missing_rgb_labels = []
    missing_semantic_segmentation_labels = []
    
    for sensor_type in missing_labels:
        for img_name in saved_imgs[sensor_type]:
            if img_name not in dataset_img_names:
                missing_labels[sensor_type].append(img_name)

    for sensor_type in imgs_not_saved:
        print(f'skipped {sensor_type} imgs: {len(imgs_not_saved[sensor_type])}')

    for sensor_type in missing_labels:
        print(f'missing {sensor_type} labels: {len(missing_labels[sensor_type])}')
    
def clean_dataset():
    """If there is an error during the data collection process, this can cause
    the saved image names and labels in the .json file to become out of sync. 
    This function can be used after such error to re-sync the dataset."""

    dataset_path = args.dataset_path
    
    with open(os.path.join(dataset_path, 'labels.json'), 'r') as file:
        dataset_info = json.load(file)

    data = dataset_info['data']
    names_from_json = set()
    for idx, dic in enumerate(data):
        for key in dic:
            names_from_json.add(dic[key]['img_name'])
        
    saved_imgs = glob.glob(dataset_path+'/*.png')
    for idx, img in enumerate(saved_imgs):
        saved_imgs[idx] = img.split('/')[-1]
        
    saved_imgs.sort()

    deleted_images = []

    for img_name in saved_imgs:
        if img_name not in names_from_json:
            os.remove(os.path.join(dataset_path, img_name))
            deleted_images.append(img_name)
            
def visualize_dataset():
    dataset_base_path = args.dataset_path
    
    labels_json_file_path = os.path.join(dataset_base_path, 'labels.json')
    
    with open(labels_json_file_path, 'r') as file:
        dataset_info = json.load(file)
        
    def preprocess_depth(depth_img_path):
        """ Normalize depth image and return h x w x 1 numpy array."""
        depth_img = cv2.imread(depth_img_path)
        depth_img = depth_img[:,:,2] + 256 * depth_img[:,:,1] + 256 * 256 * depth_img[:,:,0]
        depth_img = depth_img / (256 * 256 * 256 - 1)
        
        # the distribution of depth values was HEAVILY skewed towards the lower end
        # therfore we will try to improve the distribution by clipping between
        # 0 and a threshold and normalizing based on these
        
        # need to test with clip_coefficient = 2
        clip_coefficient = 4

        depth_img = np.clip(depth_img, 0, 1/clip_coefficient)

        depth_img = depth_img * clip_coefficient

        return np.expand_dims(depth_img, axis=-1)
        
    data = dataset_info['data']
    random.shuffle(data)
    for group_info in data:
        depth_full_img = None
        rgb_full_img = None
        for key in group_info:
            if 'depth' in key:
                # depth_img = cv2.imread(os.path.join(args.dataset_path,
                #                           group_info[key]['img_name']))
                depth_img = preprocess_depth(os.path.join(args.dataset_path,
                                          group_info[key]['img_name']))
                if depth_full_img is None:
                    depth_full_img = depth_img
                else:
                    depth_full_img = np.concatenate((depth_full_img, depth_img), axis=1)
            else:
                rgb_img = cv2.imread(os.path.join(args.dataset_path,
                                            group_info[key]['img_name']))
                if rgb_full_img is None:
                    rgb_full_img = rgb_img
                else:
                    rgb_full_img = np.concatenate((rgb_full_img, rgb_img), axis=1)
                print(group_info[key]['location'])
                
        cv2.imshow('rgb_img', rgb_full_img)
        cv2.imshow('depth_img', depth_full_img)
        cv2.waitKey(0)
        
def visualize_occlusions(args):
    """ This is used to visualize the images that contain the closes objects in 
    them. This can show if the dataset is having occlusion problems."""
    
    def preprocess_depth(depth_img_path):
        """ Normalize depth image and return h x w x 1 numpy array."""
        depth_img = cv2.imread(depth_img_path)
        depth_img = depth_img[:,:,2] + 256 * depth_img[:,:,1] + 256 * 256 * depth_img[:,:,0]
        depth_img = depth_img / (256 * 256 * 256 - 1)
        
        return depth_img
    
    img_paths = glob.glob(args.dataset_path+'/*depth*')
    img_paths.sort()
    results = []
    for idx, depth_img_path in enumerate(img_paths):
        depth_img = preprocess_depth(depth_img_path)
        results.append({
            'img': depth_img_path,
            'min_value': np.min(depth_img)
        })

    results = sorted(results, key=lambda x: x['min_value'])

    rgb_img_paths = glob.glob(args.dataset_path+'/*rgb*')
    for i in results:
        # get number
        depth_img_name = i['img'].split('/')[-1].split('.')[0]
        num = depth_img_name[-6:]
        for rgb_img_path in rgb_img_paths:
            if num in rgb_img_path:
                rgb_img = cv2.imread(rgb_img_path)
                print(i['min_value'])
                cv2.imshow('test', rgb_img)
                cv2.waitKey(0)
         
if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--verify_dataset", 
        action='store_true', 
        help='Run dataset verification')
    
    parser.add_argument(
        "--visualize_dataset", 
        action='store_true', 
        help='Run dataset verification')
    
    parser.add_argument(
        "--visualize_occlusions",
        action="store_true",
        help="visualize the most occluded images in the dataset"
    )
    
    parser.add_argument(
        "--clean_dataset", 
        action='store_true', 
        help='Run dataset cleaning')
    
    parser.add_argument(
        '--num_frames',
        action='store',
        default=NUM_FRAMES,
        type=int,
        help='Specify number of data frames to collect')
    
    parser.add_argument(
        '--dataset_path',
        action='store',
        default=DATASET_PATH,
        type=str,
        help='Specify the path to save the dataset')
    
    parser.add_argument(
        '--world',
        action='store',
        default=None,
        type=str,
        help='Specify the world to collect data on')
    
    parser.add_argument(
        '--skip_frames',
        action='store',
        default=5,
        type=int,
        help='Specify the frame interval with which to collect the data.')
    
    args = parser.parse_args()
    
    try:
        if args.verify_dataset:
            verify_dataset()
        elif args.clean_dataset:
            clean_dataset()
        elif args.visualize_dataset:
            visualize_dataset()
        elif args.visualize_occlusions:
            visualize_occlusions(args)
        else:
            main()
        
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')