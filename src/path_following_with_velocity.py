#!/usr/bin/env python

import carla
import time
import numpy as np
import matplotlib.pyplot as plt
import sys
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from agents.navigation.global_route_planner import GlobalRoutePlanner
from algs import splinepath
from algs.misc import BoxOff

rg = np.random.default_rng()


# Connect to a running Carla server and get main pointers
if (len(sys.argv) > 1):
    HOST = sys.argv[1]
else:
    HOST = 'localhost'
PORT = 2000
client = carla.Client(HOST, PORT)
client.set_timeout(2.0)

try:
    print("Connected to CARLA server version: {}".format(client.get_server_version()))
except:
    print('Not connected to server, exiting...')
    exit()


world = client.get_world()

actors = world.get_actors()
for a in actors:
    if not a.type_id == "spectator":
        a.destroy()

bpl = world.get_blueprint_library()
sp = world.get_spectator()

weather = world.get_weather()
world.set_weather(weather.ClearNoon)

# A simple first test-case -- path following in an empty world
p1 = carla.Location(x=200, y=-6, z=1)  # Start point
p2 = carla.Location(x=142.1, y=64, z=1)  # End point
p_obs_car = carla.Location(x=40,y=-4,z=0.2)
bp = rg.choice(bpl.filter('vehicle.tesla.model3'))
bp_obs = rg.choice(bpl.filter('vehicle.tesla.model3'))

actors = []
car = world.spawn_actor(bp, carla.Transform(p1))



actors.append(car)
obs_car = world.spawn_actor(bp_obs, carla.Transform(p_obs_car))
t = obs_car.get_transform()
t.location = p_obs_car
t.rotation.yaw = 180
obs_car.set_transform(t)
actors.append(obs_car)
obs_car.apply_control(carla.VehicleControl(throttle=0.0, steer=0))
print('Added a car to the world: {}'.format(car.id))

# Plan a mission

dao = GlobalRoutePlannerDAO(world.get_map(), sampling_resolution=1)
rp = GlobalRoutePlanner(dao)
rp.setup()
route = rp.trace_route(p1, p2)

# Create path object using nodes
p = np.array([(ri[0].transform.location.x, ri[0].transform.location.y)
              for ri in route])


path = splinepath.SplinePath(p, min_grid=3)
s = np.linspace(0, path.length, 500)

print('Planned a mission of length {:.2f} m'.format(path.length))

print('Position car ans spectator to beginning of path')
t = route[0][0].transform
car.set_transform(t)
t.location.z += 50  # 15 meters above car
sp.set_transform(t)

# Plot the plan in the Carla simulator.
print('Plot the plan in the CARLA simulator')
T = 10  # Time before line dissapears, negative for never
for s1, s2 in zip(s[:-1], s[1:]):
    s1_loc = carla.Location(x=float(path.x(s1)), y=float(path.y(s1)), z=0.5)
    s2_loc = carla.Location(x=float(path.x(s2)), y=float(path.y(s2)), z=0.5)
    world.debug.draw_line(s1_loc, s2_loc, thickness=0.35,
                            color=carla.Color(b=255))


class StateFeedbackController:
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

    def obsticle_distance_error(self,car_state):
        """Get distance to an obsticle to regulate speed"""
        x,y,theta,v = car_state
        obs_x= obs_car.get_transform().location.x
        return abs(obs_x - x)

    def u(self, t, w):
        def glob_stab_fact(x):
            """Series expansion of sin(x)/x around x=0."""
            return 1 - x**2/6 + x**4/120 - x**6/5040


        x, y, theta, v = w

        a = 0.7
        b = 0
        obsticle_dictance_error = self.obsticle_distance_error(w)
        if obsticle_dictance_error < v*2 or obsticle_dictance_error < 8:
            a=0.0
            b=1.0
        elif obsticle_dictance_error < v*3:
            a = 0.7 -8/(obsticle_dictance_error)
            #b = 1 - obsticle_dictance_error

        self.w.append(w)
        p_car = w[0:2]
        si, d = self.plan.project(p_car, self.s0, ds=2, s_lim=20)
        self.s0 = si

        heading = np.array([np.cos(theta), np.sin(theta)])
        theta_e = self.heading_error(heading, si)

        # No feed-forward term
        u =  - self.K.dot(np.array([d*glob_stab_fact(theta_e), theta_e]))[0]
        delta = np.max((-1.0, np.min((1.0, self.L*u))))
        #a=a*0.008
        a= np.max((0.0,np.min((0.7,a))))
        b= np.max((0.0,np.min((1.0,b))))
        self.d.append(d)
        self.delta.append(delta)
        self.s_p.append(si)
        self.theta_e.append(theta_e)
        self.t.append(t)

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


init_transform = route[0][0].transform # Set car in initial position of the plan
car.set_transform(init_transform)
car.apply_control(carla.VehicleControl(throttle=0, steer=0))

K = np.array([0.1, 0.25]).reshape((1, 2))
L = 3.5
ctrl = StateFeedbackController(K, L, path)
car_states = []
tl = []
print('Pause for a second to let the car settle')
time.sleep(1)
print('Start driving ...')
while ctrl.s0 < path.length-5:
    tck = world.wait_for_tick(1)
    t = car.get_transform()
    v = car.get_velocity()
    v = np.sqrt(v.x**2+v.y**2+v.z**2)
    w = np.array([t.location.x, t.location.y, t.rotation.yaw*np.pi/180.0, v])
    car_states.append(w)


    t.rotation = (carla.Rotation(-90,0,0))
    spt = sp.get_transform()
    t.location.z = spt.location.z
    sp.set_transform(t)
    # Compute control signal and apply to car
    u = ctrl.u(tck.timestamp.elapsed_seconds, w)
    print(v)
    car.apply_control(carla.VehicleControl(throttle=u[1], steer=u[0], brake=u[2]))

# Stop car after finished route
car.apply_control(carla.VehicleControl(throttle=0, steer=0))
car_states = np.array(car_states)
ctrl.t = np.array(ctrl.t)-ctrl.t[0]

print('Finished mission, removing car from simulator')
for a in actors:
    a.destroy()

print('Plot some results')
s = np.linspace(0, path.length, 500)
plt.figure(20, clear=True)
plt.plot(path.x(s), path.y(s), 'b', label='Planned path')
plt.plot(car_states[:, 0], car_states[:, 1], 'r', label='Actual path')
plt.xlabel('t [s]')
BoxOff()

plt.figure(21, clear=True)
plt.plot(ctrl.t, car_states[:, 3]*3.6)
plt.ylabel('km/h')
plt.title('Speed')
plt.xlabel('t [s]')
BoxOff()

plt.figure(22, clear=True)
plt.plot(ctrl.t, ctrl.delta)
plt.ylabel('%')
plt.title('Steer action')
plt.xlabel('t [s]')
BoxOff()
plt.show()
