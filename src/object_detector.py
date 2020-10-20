import carla
import numpy as np

from world_grid import WorldGrid

WORLD_GRID_RESOLUTION = .5
WORLD_GRID_THRESHOLD = 1

class ObjectDetector:
    def __init__(self, lidar, clear_interval):
        self._lidar = lidar
        self._world_grid = WorldGrid(WORLD_GRID_RESOLUTION, WORLD_GRID_THRESHOLD)
        self._clear_interval = clear_interval
        self._time_since_clear = 0

    def get_world_grid(self, delta_time):

        # Clear if enough time has passed
        self._time_since_clear += delta_time
        if self._time_since_clear >= self._clear_interval:
            self._time_since_clear -= self._clear_interval
            self._world_grid.clear()

        # Update world grid with new lidar points
        points = self._lidar.fetch_points()
        for p in points:
            self._world_grid.add_point((p.x, p.y))

        # Module output
        return self._world_grid
