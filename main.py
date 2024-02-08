# Velocity verlet
import math
import sys
import sysconfig
import time

import torch as torch
from numba import jit
import torch

import matplotlib
# import numpy as np
from matplotlib.animation import PillowWriter
from matplotlib.patches import Circle
from matplotlib import pyplot as plt, animation

from flameParticle import flameParticle

matplotlib.use('TkAgg')
torch.manual_seed(2024)
print(sysconfig.get_paths()["purelib"])  # where python look for packages
sys.path.append('C:/Python311/Lib/site-packages')
if torch.cuda.is_available():
    device = torch.device('cuda')  # GPU available
    print("CUDA is available! GPU will be used.")
else:
    device = torch.device('cpu')  # No GPU available, fallback to CPU
    print("CUDA is not available. CPU will be used.")
device = torch.device('cpu')


class BoundingBox:
    def __init__(self, particle):
        self.particle = particle
        self.min_x, self.max_x = particle.new_pos[0] - particle.particleRadius, particle.new_pos[
            0] + particle.particleRadius
        self.min_y, self.max_y = particle.new_pos[1] - particle.particleRadius, particle.new_pos[
            1] + particle.particleRadius


def sweep_and_prune(particles):
    bounding_boxes = [BoundingBox(p) for p in particles]
    sorted_boxes_x = sorted(bounding_boxes, key=lambda box: box.min_x)
    sorted_boxes_y = sorted(bounding_boxes, key=lambda box: box.min_y)
    return sorted_boxes_x, sorted_boxes_y


def check_potential_collisions(sorted_boxes_x, sorted_boxes_y):
    potential_collisions = []
    for i, box_x in enumerate(sorted_boxes_x):
        for j in range(i + 1, len(sorted_boxes_x)):
            box_y = sorted_boxes_x[j]
            if box_x.max_x < box_y.min_x:
                break
            if box_x.max_y >= box_y.min_y and box_x.min_y <= box_y.max_y:
                if box_x.particle != box_y.particle:
                    potential_collisions.append((box_x.particle, box_y.particle))

    for i, box_y in enumerate(sorted_boxes_y):
        for j in range(i + 1, len(sorted_boxes_y)):
            box_x = sorted_boxes_y[j]
            if box_y.max_y < box_x.min_y:
                break
            if box_y.max_x >= box_x.min_x and box_y.min_x <= box_x.max_x:
                if box_y.particle != box_x.particle and (box_y.particle, box_x.particle) not in potential_collisions:
                    potential_collisions.append((box_y.particle, box_x.particle))

    return potential_collisions


start = time.time()
stop_sim = 0
radius = 0.01
num_particles = 25
no_frames = 500
avg_velocity_propane = 810.  # m/ s
start_pos = torch.tensor([0.5, 0.02], device=device)
start_vel = torch.tensor([1., avg_velocity_propane ], device=device)
start_acc = torch.tensor([0., 0.], device=device)
start_force = torch.tensor([0., 0.], device=device)
particles = []
id = 0
for _ in range(num_particles):
    temperature = 12000.  # np.random.uniform(5000, 40000)
    avg_tvelocity = 50  # m/ s
    start_p = start_pos  # ( #+torch.rand(2, device=device)
    start_p[0] = start_p[0] + start_p[0] * (torch.rand(1, device=device) * 2 - 1) * 0.1
    start_v = start_vel + start_vel * (torch.rand(2, device=device) * 2 - 1) * 0.1  # - 1  # Random velocity between -5 and 5
    particle = flameParticle(start_p, start_v, start_acc, start_force, radius, temperature, num_particles + 1, device)
    # particle.temp2vel_rms()
    particle.vel_rms2temp()
    particle.id = id
    id += 1
    particle.getColorfromTemperature()
    particles.append(particle)

# particle = flameParticle(torch.tensor([0.5, 0.5]), torch.tensor([0.0, -avg_velocity_propane]), start_acc, start_force, radius * 5,
#                          temperature, num_particles + 1, device)
# # particle.temp2vel_rms()
# particle.vel_rms2temp()
# particle.id = id
# particle.getColorfromTemperature()
# particle.mass = 1e2
# particles.append(particle)

boundary_min = 0.
boundary_max = 1.
frame_data = []
sim_data = []
pprim_colides = -1
for i in range(no_frames):
    frame_start = time.time()
    frame_data = []
    for p in particles:
        dt = i * p.time_interval
        p.dt = dt

        p.get_position()
        p.get_acceleration()
        p.get_velocity()
        p.boundaryCollision(boundary_min, boundary_max)
        p.vel += p.grav_acc

        sorted_boxes_x, sorted_boxes_y = sweep_and_prune(particles)
        potential_collisions = check_potential_collisions(sorted_boxes_x, sorted_boxes_y)
        # print(potential_collisions)
        if potential_collisions is not None:
            for pair in potential_collisions:
                p_index = -1
                if pair[0] == p:
                    p_index = 1
                elif pair[1] == p:
                    p_index = 0
                if p_index != -1:
                    pcollision = p.particleCollision(pair[p_index])
                    if pcollision is not None:
                        p.vel_rms2temp()
                        p.getColorfromTemperature()
                        pcollision.vel_rms2temp()
                        pcollision.getColorfromTemperature()
                        particles[particles.index(pair[p_index])] = pcollision

        frame_data.append([p.new_pos[0].item(), p.new_pos[1].item(), p.particleRadius, p.particleColor])
    sim_data.append(frame_data)

    frame_end = time.time()
    # print('Loop time:', round((frame_end - frame_start), 3), '[s]',' iter: ',i)

end = time.time()
print('Simulation Time : ', round(end - start, 3), ' [s]')
print('FPS : ', no_frames/round(end - start, 3), ' [s]')
draw_start = time.time()
fig = plt.figure(figsize=(6, 6))
plt.style.use('dark_background')
grid = plt.GridSpec(20, 20, wspace=2, hspace=0.6)
# ax = fig.add_subplot(grid[:, :5])
ax = fig.add_subplot(grid[:, :])
ims = []

for i in range(len(sim_data)):
    pf = []
    for j in range(len(sim_data[i])):
        # Create Circle patches for each particle in the current frame
        pp = Circle((sim_data[i][j][0], sim_data[i][j][1]), sim_data[i][j][2], color=sim_data[i][j][3])
        pf.append(pp)
        ax.add_patch(pp)
    # Append the list of patches for the current frame to ims
    ims.append(pf)

ani = animation.ArtistAnimation(fig, ims, interval=30, blit=True, repeat=True)
draw_stop = time.time()
print('Drawing Time : ', round(draw_stop - draw_start, 3), ' [s]')

# ani.save("particles_diffiusion2.gif", dpi=300, writer=PillowWriter(fps=25))

# plt.grid()
plt.show()
