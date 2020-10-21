import carla
import numpy as np

LOOK_AHEAD_HORIZON = 20
LINE_RESOLUTION = .5
MAX_THROTTLE = .7
EMERGENCY_BRAKE_DISTANCE = 5
K_THROTTLE = 8
class Controller:
    def __init__(self, K, L, path=None, goal_tol=1):
        self.plan = path
        self.K = K
        self.goal_tol = goal_tol
        self.d = []
        self.delta = []
        self.theta_e = []
        self.s_p = []
        self.L = L
        self.s0 = 0
        self.t = []
        self.w = []

    def heading_error(self, heading, s):
        """Compute theta error"""
        heading0, nc = self.plan.heading(s)
        cos_alpha = heading.dot(heading0)
        sin_alpha = np.float(np.cross(heading0, heading))

        theta_e = np.arctan2(sin_alpha, cos_alpha)
        return theta_e

    def vector_to_points(self, v, origin, res):

        points = [origin]
        while np.linalg.norm(points[-1] - (origin + v)) >= res:
            points.append(points[-1] + v / np.linalg.norm(v) * res)

        return points


    def find_nearest_obstacle(self, w, wg, _world):
        origin = np.array([w[0], w[1]])
        heading = np.array([np.cos(w[2]), np.sin(w[2])])
        normal = np.array([np.sin(w[2]), -np.cos(w[2])])

        lines = []
        for i in range(-3, 4):
            lines.append(self.vector_to_points(heading * LOOK_AHEAD_HORIZON,
                                               origin + normal * i
                                               * LINE_RESOLUTION,
                                               LINE_RESOLUTION))

        closest_pos = None
        closest_dist = np.float('inf')
        for points in lines:
            for p in points:
                d = np.linalg.norm(p - origin)
                if wg.is_obstructed(p) and d < closest_dist:
                    closest_dist = d
                    closest_pos = p

                    # debug
                    _world.debug.draw_line(carla.Location(x=w[0], y=w[1], z=1),
                                           carla.Location(x=p[0], y=p[1], z=1),
                                           color=carla.Color(255, 0, 255),
                                           thickness=.07, life_time=.2)

        return closest_pos

    def u(self, t, w, wg, _world):
        def glob_stab_fact(x):
            """Series expansion of sin(x)/x around x=0."""
            return 1 - x**2/6 + x**4/120 - x**6/5040

        # Get state parameters
        x, y, theta, v = w

        # Find nearest obstacle
        obstacle_pos = self.find_nearest_obstacle(w, wg, _world)

        # Colission avoidance
        a = MAX_THROTTLE
        b = 0
        if obstacle_pos is not None:
            obstacle_dictance = np.linalg.norm(np.array([x, y]) - obstacle_pos)
            if obstacle_dictance < v * 2 or obstacle_dictance < EMERGENCY_BRAKE_DISTANCE:
                a=0.0
                b=1.0
            if obstacle_dictance < v * 3:
                #a = MAX_THROTTLE - K_THROTTLE/obstacle_dictance
                a = 0
                b = K_THROTTLE/obstacle_dictance
                print(b)
            a = np.max((0.0,np.min((MAX_THROTTLE,a))))
            b = np.max((0.0,np.min((1.0,b))))

        # Calculate position and distance error
        self.w.append(w)
        p_car = w[0:2]
        si, d = self.plan.project(p_car, self.s0, ds=2, s_lim=20)

        # Calculate rotational error and steering angle
        self.s0 = si
        heading = np.array([np.cos(theta), np.sin(theta)])
        theta_e = self.heading_error(heading, si)

        u =  - self.K.dot(np.array([d*glob_stab_fact(theta_e), theta_e]))[0]
        delta = np.max((-1.0, np.min((1.0, self.L*u))))

        # Update self
        self.d.append(d)
        self.delta.append(delta)
        self.s_p.append(si)
        self.theta_e.append(theta_e)
        self.t.append(t)

        # Output
        return np.array([delta, a, b])

    def run(self, t, w):
        p_goal = self.plan.path[-1, :]
        p_car = w[0:2]
        dp = p_car - p_goal
        dist = np.sqrt(dp.dot(dp))
        if dist < self.goal_tol:
            return False
        else:
            return True
