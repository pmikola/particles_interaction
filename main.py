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

from QuadTree import Quadtree, Box
from flameParticles import flameParticles
from sweepAndprune import sweep_and_prune, check_potential_collisions

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

start = time.time()
stop_sim = 0
num_particles = 20
no_frames = 200
avg_velocity_propane = 810.  # m/ s
dimensions = 2


def initTensor(dim, num_particles, fill_value=0., multiple_X=1., multiple_Y=1., offset_X=0., offset_Y=0., scale=2.,
               noise_amp_x=0.0,
               noise_amp_y=0.0, device=device):
    tensor = torch.full((dim, num_particles), fill_value, device=device)

    tensor[0] *= multiple_X
    tensor[0] += (torch.rand(num_particles, device=device) * scale - scale / 2.) * noise_amp_x - offset_X

    if dim > 1:
        tensor[1] *= multiple_Y
        tensor[1] += (torch.rand(num_particles, device=device) * scale - scale / 2.) * noise_amp_y - offset_Y

    return tensor


start_pos = initTensor(dim=dimensions, num_particles=num_particles, fill_value=1.,
                       multiple_X=1., multiple_Y=1.,
                       offset_X=0.5, offset_Y=0.9, scale=2.,
                       noise_amp_x=0.1, noise_amp_y=0.1, device=device)

start_vel = initTensor(dim=dimensions, num_particles=num_particles, fill_value=1.,
                       multiple_X=1., multiple_Y=avg_velocity_propane,
                       offset_X=0., offset_Y=0.0, scale=2.,
                       noise_amp_x=120., noise_amp_y=0.5, device=device)
start_acc = initTensor(dim=dimensions, num_particles=num_particles, device=device)
start_forces = initTensor(dim=dimensions, num_particles=num_particles, device=device)
temperatures = initTensor(dim=1, num_particles=num_particles, fill_value=10000., noise_amp_x=5000, device=device)
pradius = initTensor(dim=dimensions, num_particles=num_particles, fill_value=0.02, noise_amp_x=0.0, noise_amp_y=0.0,
                     device=device)

particles = flameParticles(start_pos, start_vel, start_acc, start_forces, pradius, temperatures, num_particles, device)
particles.vel_rms2temp()
particles.getColorfromTemperature()

boundary_min = 0.
boundary_max = 1.
frame_data = []
sim_data = []
pprim_colides = -1

for i in range(no_frames):
    frame_start = time.time()
    frame_data = []
    dt = i * particles.time_interval
    particles.dt = dt
    particles.get_position()
    particles.get_acceleration()
    particles.get_velocity()
    particles.boundaryCollision(boundary_min, boundary_max)
    particles.vel += particles.grav_acc.view(-1, 1)
    pbox = Box(0.0, 0.0, boundary_max, boundary_max)
    quadtree = Quadtree(pbox, 5, device)
    for idx, (x, y) in enumerate(zip(particles.pos[0], particles.pos[1])):
        quadtree.insert(x, y)
        potential_collisions = quadtree.query(pbox)
        print(potential_collisions[:])
        time.sleep(1)
        for other_idx in range(len(potential_collisions)):
            if potential_collisions[other_idx] != idx:
                particles.particleCollision(idx, other_idx)
                # particles.vel_rms2temp()
                # particles.getColorfromTemperature()

    particles.vel_rms2temp()
    particles.getColorfromTemperature()

    # sorted_boxes_x, sorted_boxes_y = sweep_and_prune(particles)
    # potential_collisions = check_potential_collisions(sorted_boxes_x, sorted_boxes_y)
    # print(potential_collisions)

    # TODO: CHANGE SWEEP AND PRUNE TO QUADTREE ALGORITHM
    # if potential_collisions is not None:
    #     for pair in potential_collisions:
    #         p_index = -1
    #         if pair[0] == p:
    #             p_index = 1
    #         elif pair[1] == p:
    #             p_index = 0
    #         if p_index != -1:
    #             pcollision = p.particleCollision(pair[p_index])
    #             if pcollision is not None:
    #                 p.vel_rms2temp()
    #                 p.getColorfromTemperature()
    #                 pcollision.vel_rms2temp()
    #                 pcollision.getColorfromTemperature()
    #                 particles[particles.index(pair[p_index])] = pcollision

    sim_data.append([particles.new_pos[0], particles.new_pos[1], particles.particleRadius, particles.particleColor])

    frame_end = time.time()
    # print('Loop time:', round((frame_end - frame_start), 3), '[s]',' iter: ',i)

end = time.time()
print('Simulation Time : ', round(end - start, 3), ' [s]')
print('FPS : ', round(no_frames / (end - start), 2), ' [fps]')
draw_start = time.time()
fig = plt.figure(figsize=(6, 6))
plt.style.use('dark_background')
grid = plt.GridSpec(20, 20, wspace=2, hspace=0.6)
# ax = fig.add_subplot(grid[:, :5])
ax = fig.add_subplot(grid[:, :])
ims = []

for i in range(no_frames):
    pf = []
    for j in range(num_particles):
        pos_x = sim_data[i][0][j].cpu().detach().numpy()
        pos_y = sim_data[i][1][j].cpu().detach().numpy()
        radi = sim_data[i][2][0][j].cpu().detach().numpy()

        colors = []
        for k in range(0, 4):
            col = sim_data[i][3][k].cpu().detach().numpy()
            colors.append(col[j])
        #     # Create Circle patches for each particle in the current frame
        pp = Circle((pos_x, pos_y), radi, color=colors)
        pf.append(pp)
        ax.add_patch(pp)
    # Append the list of patches for the current frame to ims
    ims.append(pf)

ani = animation.ArtistAnimation(fig, ims, interval=30, blit=True, repeat=True)
draw_stop = time.time()
print('Drawing Time : ', round(draw_stop - draw_start, 3), ' [s]')

ani.save("particles_diffiusion2.gif", dpi=300, writer=PillowWriter(fps=25))

# plt.grid()
plt.show()
