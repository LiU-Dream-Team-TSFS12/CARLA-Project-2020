
import carla
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from agents.navigation.global_route_planner import GlobalRoutePlanner
from algs import splinepath
from algs.misc import BoxOff

try:
    #Random number generator
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

    #Destoy all actors if there are any
    for actor in world.get_actors():
        if actor.type_id != 'spectator':
            actor.destroy()

    bpl = world.get_blueprint_library()
    sp = world.get_spectator()

    weather = world.get_weather()
    world.set_weather(weather.ClearNoon)

    #Set start and end point of route
    p1 = carla.Location(x=243, y=120, z=1)  # Start point
    p2 = carla.Location(x=248, y=-60, z=1)  # End point

    #Get pointer to a random Model3 Tesla
    bp = rg.choice(bpl.filter('vehicle.tesla.model3'))

    #Spawn a car and add it to the list of actors
    actors = []
    car = world.spawn_actor(bp, carla.Transform(p1))
    actors.append(car)

    # Plan a preliminary route (without regard to traffic
    dao = GlobalRoutePlannerDAO(world.get_map(), sampling_resolution=1)
    rp = GlobalRoutePlanner(dao)
    rp.setup()
    route = rp.trace_route(p1, p2)

    # Spawn an obstacle car on the planned route
    bp = rg.choice(bpl.filter('vehicle.audi.*'))
    obs = world.spawn_actor(bp, route[40][0].transform)
    #Make obstacle car drive slowly
    obs.apply_control(carla.VehicleControl(throttle=0.3, steer=0))
    actors.append(obs)

    #Initialize car and spectator
    t = route[0][0].transform
    car.set_transform(t)
    t.location.z += 25  # 15 meters above car
    t.rotation = (carla.Rotation(-90,0,0))
    sp.set_transform(t)
    car.apply_control(carla.VehicleControl(throttle=0, steer=0))

    #Set controll parameters
    K = np.array([.01, 0.15]).reshape((1, 2))
    L = 3.5

    #Wait for car to be ready
    time.sleep(1)

    #Import and create instances of modules
    from lidar import Lidar
    lidar = Lidar(world, car)

    from object_detector import ObjectDetector
    object_detector = ObjectDetector(lidar, 1)

    from route_planner import RoutePlanner
    route_planner = RoutePlanner([r[0] for r in route], .5)

    lidar.start()

    from controller import Controller
    ctrl = Controller(K, L)


    previous_time = -1
    while True:
        tck = world.wait_for_tick(1)
        t = car.get_transform()
        v = car.get_velocity()
        v = np.sqrt(v.x**2+v.y**2+v.z**2)
        w = np.array([t.location.x, t.location.y, t.rotation.yaw*np.pi/180.0, v])

        spt = sp.get_transform()
        spt.location = t.location
        spt.location.z += 80
        sp.set_transform(spt)

        if previous_time == -1:
            previous_time = tck.timestamp.elapsed_seconds

        dt = tck.timestamp.elapsed_seconds - previous_time
        wg = object_detector.get_world_grid(dt)
        path = route_planner.get_path(dt, w, wg, world)

        #If car is on the end of the path
        if path is None:
            break

        # Temporary debug draw
        """
        s = np.linspace(0, path.length, 500)
        for s1, s2 in zip(s[:-1], s[1:]):
            s1_loc = carla.Location(x=float(path.x(s1)), y=float(path.y(s1)), z=0.5)
            s2_loc = carla.Location(x=float(path.x(s2)), y=float(path.y(s2)), z=0.5)
            world.debug.draw_line(s1_loc, s2_loc, thickness=0.35,
                                  life_time=.1, color=carla.Color(b=255))
        """
        # Compute control signal and apply to car
        u = ctrl.u(tck.timestamp.elapsed_seconds, w, wg, path, world)
        car.apply_control(carla.VehicleControl(throttle=u[1], steer=u[0], brake=u[2]))


except KeyboardInterrupt:
    pass


finally:
    # Stop car after finished route
    lidar.stop()
    car.apply_control(carla.VehicleControl(throttle=0, steer=0))
    obs.apply_control(carla.VehicleControl(throttle=0, steer=0))
    ctrl.t = np.array(ctrl.t)-ctrl.t[0]
    #Destoy all actors if there are any
    """
    for actor in world.get_actors():
        if actor.type_id != 'spectator':
            actor.destroy()
    """
