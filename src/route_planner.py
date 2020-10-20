class RoutePlanner:
    def __init__(self, waypoints, clear_interval):
        self._waypoints = waypoints
        self._time = 0
        self._clear_interval = clear_interval
        self._prev_s = 0

        #Plan a splinepath
        p = np.array([(wp.transform.location.x, wp.transform.location.y)
                      for wp in waypoints])
        path = splinepath.SplinePath(p, min_grid=3)
        self._path = path

    def get_path(self, dt, w, wg, _world):
        self._time += dt

        if self._time < self._clear_interval:
            return path

        self._time -= self._clear_interval

        #Calculate current state
        x,y,theta,v = w
        p = np.array([x,y])
        s0 = self._path.project(p, self._prev_s, .5, 10)[0]
        self._prev_s = s0

        # Remove past waypoints
        _, wp_i = get_prev_waypoint(self._path.p(s0))
        self._waypoints = self._waypoints[wp_i:]

        #Calculate obstruction point
        collision_point = _get_path_obstruction_point(s0, 20, .5, wg)
        wp_c, wp_ci = get_prev_waypoint(collision_point)

        for wp in waypoints:
            _world.debug.draw_point(carla.Location(x=wp.transform.location.x, y=wp.transform.location.y, z=.8),
                                    color=(carla.Color(0, 255, 0) if wp == wp_c else carla.Color(255, 0, 0)),
                                    size=.2, life_time=1.5)

        return self._path





    def _get_path_obstruction_point(self, s0, d, res, wg):
        for s in range(s0, s0+d, res):
            p = self._path.p(s)
            if wg.is_obstructed(p):
                return p
            return -1.0

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
            if prev_dist_vect is not None and np.dot(pre_dist_vect, dist_vect) < 0:
                return self._waypoints[i-1], i-1

            prev_dist_vect = dist_vect

        return self._waypoints[-1], len(self._waypoints)-1
