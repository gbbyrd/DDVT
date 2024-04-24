import carla
import queue
import random

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
        
        self.FRONT_X = kwargs.get('front_x', 2.5)
        self.FRONT_Z = kwargs.get('front_z', 2.5)
        
        self.low_sensors = []
        self.low_sensor_info = []
        
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
        
        self.randomize_low_sensors()
        
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
        for low_sensor in self.low_sensors:
            low_sensor.destroy()
        for actor in self.actors:
            actor.destroy()
        
    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data       