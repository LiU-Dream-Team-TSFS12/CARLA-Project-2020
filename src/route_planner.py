import numpy as np
import carla
from algs import splinepath

NUM_WAYPOINTS_PER_LANE_CHANGE = 15
NUM_WAYPOINTS_PER_OVERTAKE = 10

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
        return splinepath.SplinePath(p, min_grid=3)

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
        collision_point = self._get_path_obstruction_point(self._path, s0, 20, .1, wg)

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

        # Collision was found, find closest safe waypoint
        wp_c, wp_ci = self._get_prev_waypoint(collision_point)

        print("--------------------")
        print("Collision was found!")

        # ##############

        # Update waypoints

        lane_change_start_index = max(wp_ci - NUM_WAYPOINTS_PER_LANE_CHANGE, 0)
        lane_change_done_index = wp_ci
        lane_overtake_done_index = lane_change_done_index + NUM_WAYPOINTS_PER_OVERTAKE
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

            if not (wp.lane_change in (carla.LaneChange.Left, carla.LaneChange.Both) and not wp.is_junction):
                # Could not change lane
                self._path = self._gen_spline_path(self._waypoints)
                return self._path

            new_waypoints.append(wp.get_left_lane())

        # Remaining waypoints (after full overtake)
        new_waypoints += self._waypoints[lane_rejoin_index + 1:]

        for i, wp in enumerate(new_waypoints):
            _world.debug.draw_point(wp.transform.location,
                                    color=carla.Color(0, 255, 255),
                                    size=.1, life_time=20)

        # Generate new path
        new_path = self._gen_spline_path(new_waypoints)

        """s = np.linspace(0, new_path.length, 500)
        for s1, s2 in zip(s[:-1], s[1:]):
            s1_loc = carla.Location(x=float(new_path.x(s1)), y=float(new_path.y(s1)), z=0.5)
            s2_loc = carla.Location(x=float(new_path.x(s2)), y=float(new_path.y(s2)), z=0.5)
            _world.debug.draw_line(s1_loc, s2_loc, thickness=0.35,
                                  life_time=.5, color=carla.Color(r=255))"""

        # Check if new path has collisions
        if self._get_path_obstruction_point(new_path, 0, np.linalg.norm(p - np.array([wp_c.transform.location.x, wp_c.transform.location.y])), .1, wg) is None:
            self._waypoints = new_waypoints
            self._path = new_path

            return self._path

        # Could not find a sufficient correction path
        self._path = self._gen_spline_path(self._waypoints)
        return self._path

    def _get_path_obstruction_point(self, path, s0, d, res, wg):
        for s in np.linspace(s0, s0+d, d / res + 1):
            p = path.p(s)
            """_world.debug.draw_point(carla.Location(x=p[0], y=p[1], z=.9),
                                    color=carla.Color(0, 0, 255),
                                    size=.1, life_time=.4)"""
            if wg.is_obstructed(p):
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
            if prev_dist_vect is not None and np.dot(prev_dist_vect, dist_vect) < 0:
                return self._waypoints[i-1], i-1

            prev_dist_vect = dist_vect
        return self._waypoints[0], 0
