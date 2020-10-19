import carla
import random
import matplotlib.pyplot as plt
import numpy as np

# Parse arguments
import argparse

argparser = argparse.ArgumentParser(
        description=__doc__)
argparser.add_argument(
    '--host',
    metavar='H',
    default='127.0.0.1',
    help='IP of the host server (default: 127.0.0.1)')
args = argparser.parse_args()

# Connect
HOST = args.host
PORT = 2000
client = carla.Client(HOST, PORT)
client.set_timeout(2.0)

world = client.get_world()
bpl = world.get_blueprint_library()
sp = world.get_spectator()

# Spawn car
bp = random.choice(bpl.filter('vehicle.audi.*'))
car = world.spawn_actor(bp, carla.Transform(sp.get_transform().location))

# Lidar
lidar_bpl = bpl.filter('sensor.lidar.ray_cast')[0]
lidar_bpl.set_attribute('range', '5000')
lidar = world.spawn_actor(lidar_bpl,
                  carla.Transform(carla.Location(x=2.5, z=1),
                                  carla.Rotation(pitch=0)),
                  attach_to=car)

# Loop
lidar_data = []
i = 0
#plt.figure(10, clear=True)
#plt.axis([-10, 10, -10, 10])
def lidar_callback(data):

    points = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))
    points =  np.reshape(points, (int(points.shape[0] / 4), 4))
    loc = data.transform.location
    for px, py, pz in zip(points[:,0], points[:,1], points[:,2]):
        if pz < -0.8 or pz > 1.2:
            continue
        x = loc.x + px
        y = loc.y + py
        z = loc.z + pz
        
        world.debug.draw_point(location=carla.Location(x=x, y=y, z=z), size=0.1, color=carla.Color(255,0,0), life_time=1)
    #plt.plot(points[h, 0], points[h, 1], 'k.', markersize=0.3)

input('press to start lidar')
lidar.listen(lidar_callback)


# Wait for input
input('press to stop lidar')
lidar.stop()

#plt.show()

# Exit
lidar.destroy()
car.destroy()