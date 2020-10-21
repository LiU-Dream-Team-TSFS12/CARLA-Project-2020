import numpy as np
import carla
from algs import splinepath

NUM_WAYPOINTS_PER_LANE_CHANGE = 15
NUM_WAYPOINTS_PER_OVERTAKE = 15
NUM_WAYPOINTS_PER_MARGIN = 0

WIDTH_RESOLUTION = .5

class RoutePlanner:
    def __init__(self, waypoints, clear_interval):
        self._waypoints = waypoints
        self._time = 0
        self._clear_interval = clear_interval
        self._prev_s = 0

        # Plan a splinepath
        self._path = self._gen_spline_path(waypoints)

    def _gen_spline_path(self, waypoints):
        p = np.array([(wp.transform.location.x, wp.transform.location.y)
                      for wp in waypoints])
        return splinepath.SplinePath(p, min_grid=1)

    def get_path(self, dt, w, wg, _world):
        self._time += dt

        if self._time < self._clear_interval:
            return self._path

        self._time -= self._clear_interval

        # Fetch current state
        x,y,theta,v = w
        p = np.array([x,y])
        s0 = self._path.project(p, self._prev_s, .5, 10)[0]
        self._prev_s = s0

        # Remove past waypoints
        _, wp_i = self._get_prev_waypoint(self._path.p(s0))
        self._waypoints = self._waypoints[wp_i:]

        # Calculate obstruction point
        collision_point = self._get_path_obstruction_point(self._path, s0, 30, .1, wg)

        for c in wg.get_corners():
            for i in range(len(c)):
                _world.debug.draw_line(carla.Location(x=c[i-1][0], y=c[i-1][1], z=.7),
                                       carla.Location(x=c[i][0], y=c[i][1], z=.7),
                                       thickness = .07, color=carla.Color(255, 0, 0),
                                       life_time = .4)

        # If no collision was found, return trimmed path
        if collision_point is None:
            self._path = self._gen_spline_path(self._waypoints)
            return self._path

        # Collision was found, find closest safe waypoint5
        wp_c, wp_ci = self._get_prev_waypoint(collision_point)

        print("--------------------")
        print("Collision was found!")

        # ##############

        # Update waypoints

        lane_change_start_index = max(wp_ci - NUM_WAYPOINTS_PER_LANE_CHANGE - NUM_WAYPOINTS_PER_MARGIN, 0)
        lane_change_done_index = max(wp_ci - NUM_WAYPOINTS_PER_MARGIN, 0)
        lane_overtake_done_index = lane_change_done_index + NUM_WAYPOINTS_PER_OVERTAKE + NUM_WAYPOINTS_PER_MARGIN
        lane_rejoin_index = lane_overtake_done_index + NUM_WAYPOINTS_PER_LANE_CHANGE

        # Waypoints until lane change
        new_waypoints = self._waypoints[0:lane_change_start_index + 1]

        """# Interpolate waypoints for lane change
        start_vect = np.array([new_waypoints[-1].transform.location.x,
                               new_waypoints[-1].transform.location.y)
        end_vect = np.array([self._waypoints[lane_change_done_index + 1].transform.location.x,
                             self._waypoints[lane_change_done_index + 1].transform.location.y])
        change_vect = (end_vect - start_vect) / np.linalg.norm(end_vect - start_vect)
        for i in range(NUM_WAYPOINTS_PER_LANE_CHANGE):
            new_waypoints += change"""


        # Overtake waypoints
        for i in range(lane_change_done_index + 1, lane_overtake_done_index+1):
            wp = self._waypoints[i]

            if not (wp.get_left_lane() is not None and wp.lane_change in (carla.LaneChange.Left, carla.LaneChange.Both)):
                # Could not change lane
                self._path = self._gen_spline_path(self._waypoints)
                return self._path

            new_waypoints.append(wp.get_left_lane())

        # Remaining waypoints (after full overtake)
        new_waypoints += self._waypoints[lane_rejoin_index + 1:]

        for i, wp in enumerate(new_waypoints):
            _world.debug.draw_point(wp.transform.location,
                                    color=carla.Color(0, 255, 255),
                                    size=.1, life_time=1)

        # Generate new path
        new_path = self._gen_spline_path(new_waypoints)

        # Check if new path has collisions
        if self._get_path_obstruction_point(new_path, 0, np.linalg.norm(p - np.array([wp_c.transform.location.x, wp_c.transform.location.y])), .1, wg) is None:
            self._waypoints = new_waypoints
            self._path = new_path

        return self._path

        # Could not find a sufficient correction path
        self._path = self._gen_spline_path(self._waypoints)
        return self._path

    def _get_path_obstruction_point(self, path, s0, d, res, wg):
        displacement=0
        distance = 0.0

        points=[]
        while distance < d:
            s = s0 + displacement
            x = path.p(s)[0]
            y = path.p(s)[1]
            if points:
                delta_x = points[-1][0] - x
                delta_y = points[-1][1] - y
                distance += np.linalg.norm(np.array([delta_x, delta_y]))

            points.append(np.array([x, y]))
            displacement += res

        for p in points:
            for i in [-4,-3,-2,-1,1,2,3,4]:
                normal = np.array([-p[1], p[0]])
                normal /= np.linalg.norm(normal)
                new_p = p + i*res*normal
                if wg.is_obstructed(new_p):
                    return p

        return None

    def _get_prev_waypoint(self, p):
        prev_dist_vect = None

        for i, wp in enumerate(self._waypoints):
            # Vector from wp to p
            wp_pos = np.array([wp.transform.location.x, wp.transform.location.y])
            dist_vect = p - wp_pos

            # Check if on waypoint (with tolerance)
            d = np.linalg.norm(dist_vect)
            if d < .1:
                return wp, i

            # Normalize vector
            dist_vect /= d

            # Check if waypoints passed p
            if prev_dist_vect is not None and np.dot(prev_dist_vect, dist_vect) < np.cos(np.pi / 4):
                return self._waypoints[i-1], i-1

            prev_dist_vect = dist_vect
        return self._waypoints[0], 0
