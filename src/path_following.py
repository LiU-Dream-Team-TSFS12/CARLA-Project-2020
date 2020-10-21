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

for actor in world.get_actors():
    if actor.type_id != 'spectator':
        actor.destroy()

bpl = world.get_blueprint_library()
sp = world.get_spectator()

weather = world.get_weather()
world.set_weather(weather.ClearNoon)

# A simple first test-case -- path following in an empty world
#p1 = carla.Location(x=200, y=-6, z=1)  # Start point
#p2 = carla.Location(x=142.1, y=64, z=1)  # End point
p1 = carla.Location(x=243, y=120, z=1)  # Start point
p2 = carla.Location(x=248, y=-60, z=1)  # End point

bp = rg.choice(bpl.filter('vehicle.tesla.model3'))
print('Random selection: {}'.format(bp.id))

actors = []
car = world.spawn_actor(bp, carla.Transform(p1))
actors.append(car)

print('Added a car to the world: {}'.format(car.id))

# Plan a mission

dao = GlobalRoutePlannerDAO(world.get_map(), sampling_resolution=1)
rp = GlobalRoutePlanner(dao)
rp.setup()
route = rp.trace_route(p1, p2)

# Spawn an obstacle car
bp = rg.choice(bpl.filter('vehicle.audi.*'))
obs = world.spawn_actor(bp, route[40][0].transform)
obs.apply_control(carla.VehicleControl(throttle=0.3, steer=0))
actors.append(obs)


# Create path object using nodes
p = np.array([(ri[0].transform.location.x, ri[0].transform.location.y)
              for ri in route])
path = splinepath.SplinePath(p, min_grid=3)
s = np.linspace(0, path.length, 500)

print('Planned a mission of length {:.2f} m'.format(path.length))

print('Position car ans spectator to beginning of path')
t = route[0][0].transform
car.set_transform(t)
t.location.z += 15  # 15 meters above car
sp.set_transform(t)

init_transform = route[0][0].transform # Set car in initial position of the plan
car.set_transform(init_transform)
car.apply_control(carla.VehicleControl(throttle=0, steer=0))

#K = np.array([.15, 3]).reshape((1, 2)) / 10
K = np.array([.01, 0.15]).reshape((1, 2))
L = 3.5
car_states = []
tl = []
print('Pause for a second to let ow = time.time()the car settle')
time.sleep(1)
print('Start driving ...')

from lidar import Lidar
lidar = Lidar(world, car)

from object_detector import ObjectDetector
object_detector = ObjectDetector(lidar, 1)

from route_planner import RoutePlanner
route_planner = RoutePlanner([r[0] for r in route], .1)

lidar.start()

from controller import Controller
ctrl = Controller(K, L)

import time

previous_time = -1
while True:
    tck = world.wait_for_tick(1)
    t = car.get_transform()
    v = car.get_velocity()
    v = np.sqrt(v.x**2+v.y**2+v.z**2)
    w = np.array([t.location.x, t.location.y, t.rotation.yaw*np.pi/180.0, v])
    car_states.append(w)

    if previous_time == -1:
        previous_time = tck.timestamp.elapsed_seconds

    dt = tck.timestamp.elapsed_seconds - previous_time

    #now = time.time()
    wg = object_detector.get_world_grid(dt)
    #print("obj. detector duration: ", time.time() - now)

    #now = time.time()
    path = route_planner.get_path(dt, w, wg, world)
    #print("route planner duration: ", time.time() - now)

    # Temporary debug draw
    s = np.linspace(0, path.length, 500)
    for s1, s2 in zip(s[:-1], s[1:]):
        s1_loc = carla.Location(x=float(path.x(s1)), y=float(path.y(s1)), z=0.5)
        s2_loc = carla.Location(x=float(path.x(s2)), y=float(path.y(s2)), z=0.5)
        world.debug.draw_line(s1_loc, s2_loc, thickness=0.35,
                              life_time=.5, color=carla.Color(b=255))

    # Compute control signal and apply to car
    u = ctrl.u(tck.timestamp.elapsed_seconds, w, wg, path, world)
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
