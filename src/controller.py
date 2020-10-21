import carla
import numpy as np

LOOK_AHEAD_HORIZON = 30
LINE_RESOLUTION = .5
WIDTH_RESOLUTION = .25
MAX_THROTTLE = .7
EMERGENCY_BRAKE_DISTANCE = 6
K_BRAKE = 8


class PathPoint:
    def __init__(self, x, y, dist, normal):
        self.x = x
        self.y = y
        self.dist = dist
        self.normal = normal

class Controller:
    def __init__(self, K, L, goal_tol=1):
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

    def heading_error(self, heading, s, path):
        """Compute theta error"""
        heading0, nc = path.heading(s)
        cos_alpha = heading.dot(heading0)
        sin_alpha = np.float(np.cross(heading0, heading))

        theta_e = np.arctan2(sin_alpha, cos_alpha)
        return theta_e

    def perception_points(self, origin, path):
        #points = [origin]
        #while np.linalg.norm(points[-1] - (origin + v)) >= res:
            #points.append(points[-1] + v / np.linalg.norm(v) * res)
        s0=self.s0
        #s = self.plan.project(origin,0)
        #print(self.plan.p(s0))
        displacement=0
        distance = 0.0

        points=[]
        while distance < LOOK_AHEAD_HORIZON:
            s = s0 + displacement
            x = path.p(s)[0]
            y = path.p(s)[1]
            if points:
                delta_x = points[-1].x - x
                delta_y = points[-1].y - y
                distance += np.linalg.norm(np.array([delta_x, delta_y]))

            point = PathPoint(x, y, distance, path.heading(s)[1])
            #print(plan.heading(s)[1])
            points.append(point)
            displacement += LINE_RESOLUTION

        side_points = []
        for p in points:
            for i in [-4,-3,-2,-1,1,2,3,4]:
                p_vec = np.array([p.x, p.y])
                #print(p.normal)
                new_p = p_vec + i*WIDTH_RESOLUTION*p.normal
                side_points.append(PathPoint(new_p[0], new_p[1], p.dist, p.normal))
        points += side_points
        return points



    def find_nearest_obstacle(self, w, wg, path, _world):
        origin = np.array([w[0], w[1]])

        closest_dist = float('inf')

        for p in self.perception_points(origin, path):
            d = p.dist

            if wg.is_obstructed((p.x,p.y)) and d < closest_dist:
                closest_dist = d


            # debug
            _world.debug.draw_point(carla.Location(x=p.x, y=p.y, z=1),
                                       color=carla.Color(255, 0, 255),
                                       size=.02, life_time=.2)

        return closest_dist

    def u(self, t, w, wg, path, _world):
        def glob_stab_fact(x):
            """Series expansion of sin(x)/x around x=0."""
            return 1 - x**2/6 + x**4/120 - x**6/5040

        # Get state parameters
        x, y, theta, v = w

        # Find nearest obstacle
        obstacle_distance = self.find_nearest_obstacle(w, wg, path, _world)

        # Colission avoidance
        a = MAX_THROTTLE
        b = 0
        if obstacle_distance !=  float('inf'):
            print("Closest at: %f \tm" % obstacle_distance)
            if obstacle_distance < EMERGENCY_BRAKE_DISTANCE:
                a=0.0
                b=1.0
            elif obstacle_distance > 50:
                a = MAX_THROTTLE
            elif obstacle_distance < v * 3:
                a = 0.0
                b = K_BRAKE/obstacle_distance
            else:
                a = v/14 + obstacle_distance/40
                b = 0.0

            a = np.max((0.0,np.min((MAX_THROTTLE,a))))
            b = np.max((0.0,np.min((1.0,b))))


        #print("T: %f \tB: %f" % (a, b))

        # Calculate position and distance error
        self.w.append(w)
        p_car = w[0:2]
        si, d = path.project(p_car, self.s0, ds=2, s_lim=20)

        # Calculate rotational error and steering angle
        self.s0 = si
        heading = np.array([np.cos(theta), np.sin(theta)])
        theta_e = self.heading_error(heading, si, path)

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
