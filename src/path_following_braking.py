#!/usr/bin/env python

import carla
import time
import numpy as np

import sys
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from agents.navigation.global_route_planner import GlobalRoutePlanner
from algs import splinepath
from algs.misc import BoxOff
from lidar import Lidar
from object_detector import ObjectDetector
from controller import Controller

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

# clear old actors
actors = world.get_actors()
for a in actors:
    if a.type_id == "vehicle.tesla.model3":
        a.destroy()

bpl = world.get_blueprint_library()
sp = world.get_spectator()

weather = world.get_weather()
world.set_weather(weather.ClearNoon)

# A simple first test-case -- path following in an empty world
p1 = carla.Location(x=200, y=-6, z=1)  # Start point
p2 = carla.Location(x=142.1, y=64, z=1)  # End point

bp = rg.choice(bpl.filter('vehicle.tesla.model3'))
print('Random selection: {}'.format(bp.id))

actors = []
car = world.spawn_actor(bp, carla.Transform(p1))
actors.append(car)

print('Added a car to the world: {}'.format(car.id))

# Plan a mission

dao = GlobalRoutePlannerDAO(world.get_map(), sampling_resolution=5)
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
t.location.z += 40  # 25 meters above car
sp.set_transform(t)

# Plot the plan in the Carla simulator.
print('Plot the plan in the CARLA simulator')
T = 10  # Time before line dissapears, negative for never
for s1, s2 in zip(s[:-1], s[1:]):
    s1_loc = carla.Location(x=float(path.x(s1)), y=float(path.y(s1)), z=0.5)
    s2_loc = carla.Location(x=float(path.x(s2)), y=float(path.y(s2)), z=0.5)
    world.debug.draw_line(s1_loc, s2_loc, thickness=0.35,
                          life_time=T, color=carla.Color(b=255))

init_transform = route[0][0].transform # Set car in initial position of the plan
car.set_transform(init_transform)
car.apply_control(carla.VehicleControl(throttle=0, steer=0))

K = np.array([0.1, 0.25]).reshape((1, 2))
L = 3.5
car_states = []
tl = []
print('Pause for a second to let the car settle')
time.sleep(1)
print('Start driving ...')

# Modules
lidar = Lidar(world, car)

object_detector = ObjectDetector(lidar, .1)

ctrl = Controller(K, L, world.debug)

lidar.start()

while ctrl.s0 < path.length-5:
    tck = world.wait_for_tick(1)
    t = car.get_transform()
    v = car.get_velocity()
    v = np.sqrt(v.x**2+v.y**2+v.z**2)
    w = np.array([t.location.x, t.location.y, t.rotation.yaw*np.pi/180.0, v])
    car_states.append(w)

    # Simulate modules
    wg = object_detector.get_world_grid(tck.timestamp.delta_seconds)
    u = ctrl.u(tck.timestamp.elapsed_seconds, w, wg, path)

    # Camera following
    #t.rotation = (carla.Rotation(-90,90,0))
    #spt = sp.get_transform()
    #t.location.z = spt.location.z
    #sp.set_transform(t)

    # Apply control signal
    car.apply_control(carla.VehicleControl(throttle=u[1], steer=u[0], brake=u[2]))

lidar.stop()

# Stop car after finished route
car.apply_control(carla.VehicleControl(throttle=0, steer=0))
car_states = np.array(car_states)
ctrl.t = np.array(ctrl.t)-ctrl.t[0]

print('Finished mission, removing car from simulator')
for a in actors:
    a.destroy()
