import carla
import numpy as np

from threading import Lock
from collections import deque


LIDAR_BUFFER_LIMIT = 1000

class Lidar:
    def __init__(self, world, car):
        lidar_bpl = world.get_blueprint_library().filter('sensor.lidar.ray_cast')[0]
        lidar_bpl.set_attribute('range', '5000')
        self._lidar = world.spawn_actor(lidar_bpl,
                                       carla.Transform(carla.Location(x=2.5, z=1),
                                                       carla.Rotation(pitch=0)),
                                       attach_to=car)
        self._mtx = Lock()
        self._buffer = deque([], LIDAR_BUFFER_LIMIT)

    def _callback(self, data):

        # Extract points
        points = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))
        points =  np.reshape(points, (int(points.shape[0] / 3), 3))

        # Get lidar location & rotational yaw-correction
        sen_loc = data.transform.location
        v = np.pi / 2 + data.transform.rotation.yaw * np.pi / 180

        # Add points to buffer
        self._store_points(data, sen_loc, v)

    def _store_points(self, data, sen_loc, v):
        self._mtx.acquire()
        for loc in data:
            if loc.z >= 0 or (loc.y > 0 and loc.x**2 + loc.y**2 < 2**2):
                continue
            self._buffer.appendleft(carla.Location(x = np.cos(v) * loc.x - np.sin(v) * loc.y,
                                                   y = np.sin(v) * loc.x + np.cos(v) * loc.y,
                                                   z = loc.z) + sen_loc)
        self._mtx.release()

    def fetch_points(self):
        points = []
        
        self._mtx.acquire()
        while len(self._buffer) > 0:
            points.append(self._buffer.pop())
        self._mtx.release()
        
        return points
    
    def start(self):
        def callback(data):
            self._callback(data)
        self._lidar.listen(callback)

    def stop(self):
        self._lidar.stop()
