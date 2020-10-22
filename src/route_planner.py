import numpy as np
import carla
from algs import splinepath

NUM_WAYPOINTS_PER_LANE_CHANGE = 15
NUM_WAYPOINTS_PER_OVERTAKE = 15
NUM_WAYPOINTS_PER_MARGIN = 0

RESOLUTION = .5

ON_WAYPOINT_TOLERANCE = .1

LOOK_AHEAD_HORIZON = 30
LOOK_AHEAD_RESOLUTION = .1

class RoutePlanner:
    def __init__(self, waypoints, update_interval):
        self._waypoints = waypoints
        self._time = 0
        self._update_interval = update_interval
        self._prev_s = 0

        # Plan a splinepath
        self._path = self._gen_spline_path(waypoints)


    def _gen_spline_path(self, waypoints):
        """ Generate a SplinePath based on specified waypoints. """
        p = np.array([(wp.transform.location.x, wp.transform.location.y)
                      for wp in waypoints])
        return splinepath.SplinePath(p, min_grid=1)

    def get_path(self, dt, w, wg):
        """ The RoutePlanner system function. """

        # Make sure to only update with a certain time interval
        self._time += dt
        if self._time < self._update_interval:
            return self._path
        self._time -= self._update_interval

        # Fetch current state
        x, y, theta, v = w
        p = np.array([x,y])
        s0 = self._path.project(p, self._prev_s, RESOLUTION, 10)[0]
        self._prev_s = s0

        # Check if we're done
        if s0 > self._path.length - 5:
            return None

        # Remove past waypoints
        _, wp_i = self._get_prev_waypoint(self._path.p(s0))
        self._waypoints = self._waypoints[wp_i:]

        # Calculate obstruction point
        #look_ahead = min(30, self._path.length - s0 - 1)
        collision_point = self._get_path_obstruction_point(self._path, s0,
                                                           LOOK_AHEAD_HORIZON,
                                                           LOOK_AHEAD_RESOLUTION,
                                                           wg)

        # TODO: MOVE TO ANOTHER FILE
        """for c in wg.get_corners():
            for i in range(len(c)):
                _world.debug.draw_line(carla.Location(x=c[i-1][0], y=c[i-1][1], z=.7),
                                       carla.Location(x=c[i][0], y=c[i][1], z=.7),
                                       thickness = .07, color=carla.Color(255, 0, 0),
                                       life_time = .4)"""

        # If no collision was found, return trimmed path
        if collision_point is None:
            self._path = self._gen_spline_path(self._waypoints)
            return self._path

        # Collision was found, find closest safe waypoint
        wp_c, wp_ci = self._get_prev_waypoint(collision_point)

        # Indices used for creation of correction path waypoints
        lane_change_start_index = max(wp_ci - NUM_WAYPOINTS_PER_LANE_CHANGE - NUM_WAYPOINTS_PER_MARGIN, 0)
        lane_change_done_index = max(wp_ci - NUM_WAYPOINTS_PER_MARGIN, 0)
        lane_overtake_done_index = lane_change_done_index + NUM_WAYPOINTS_PER_OVERTAKE + NUM_WAYPOINTS_PER_MARGIN
        lane_rejoin_index = lane_overtake_done_index + NUM_WAYPOINTS_PER_LANE_CHANGE

        # Waypoints until lane change
        new_waypoints = self._waypoints[0:lane_change_start_index + 1]

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

        # Generate new path
        new_path = self._gen_spline_path(new_waypoints)

        # Check if new path has collisions
        look_ahead = np.linalg.norm(p - np.array([wp_c.transform.location.x, wp_c.transform.location.y]))
        if self._get_path_obstruction_point(new_path, 0, look_ahead, .1, wg) is None:
            self._waypoints = new_waypoints
            self._path = new_path
            return self._path

        # Could not find a sufficient correction path
        self._path = self._gen_spline_path(self._waypoints)
        return self._path

    def _get_path_obstruction_point(self, path, s0, d, res, wg):
        """ Traverse a path and find a collision point on it. """

        displacement=0
        distance = 0

        # Find points along path
        points=[]
        while distance < min(path.length - s0 - 1, d):
            s = s0 + displacement
            x = path.p(s)[0]
            y = path.p(s)[1]
            if points:
                delta_x = points[-1][0] - x
                delta_y = points[-1][1] - y
                distance += np.linalg.norm(np.array([delta_x, delta_y]))

            points.append(np.array([x, y]))
            displacement += res

        # Generate points along path-normal for every point in 'points'
        for p in points:
            for i in [-4,-3,-2,-1,1,2,3,4]:
                normal = np.array([-p[1], p[0]])
                normal /= np.linalg.norm(normal)
                new_p = p + i*res*normal
                if wg.is_obstructed(new_p):
                    return p

        # There was no collision
        return None

    def _get_prev_waypoint(self, p):
        """ Find the closest waypoint behind the specified path-point. """
        prev_dist_vect = None

        for i, wp in enumerate(self._waypoints):
            # Vector from wp to p
            wp_pos = np.array([wp.transform.location.x, wp.transform.location.y])
            dist_vect = p - wp_pos

            # Check if on waypoint (with tolerance)
            d = np.linalg.norm(dist_vect)
            if d < ON_WAYPOINT_TOLERANCE:
                return wp, i

            # Normalize vector
            dist_vect /= d

            # Check if specified path-point was behind 'wp'
            if prev_dist_vect is not None and np.dot(prev_dist_vect, dist_vect) < 0:
                return self._waypoints[i-1], i-1

            # Set current waypoint as previous
            prev_dist_vect = dist_vect

        # No waypoint was found
        return self._waypoints[0], 0
