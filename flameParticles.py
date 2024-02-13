import math
import time
from numba import jit
import numpy as np
import torch
from torch import jit
from numba.experimental import jitclass


class flameParticles(object):
    def __init__(self, pos, vel, acc, forces, radius, temperature, num_particles, device):
        self.particleColor = None
        self.id = torch.range(start=0, end=num_particles - 1, step=1)
        self.device = device
        self.new_pos = torch.tensor(pos, device=self.device)
        self.new_vel = torch.tensor(vel, device=self.device)
        self.new_acc = torch.tensor(acc, device=self.device)
        self.num_of_particles = num_particles
        self.interaction_matrix = torch.zeros((self.num_of_particles, self.num_of_particles), device=self.device)
        self.time_interval = 2e-6
        self.dt = 0
        self.opacity = torch.full((1, self.num_of_particles), 1.0, device=self.device)
        self.particleRadius = radius
        self.pos = torch.tensor(pos, device=self.device)
        self.vel = torch.tensor(vel, device=self.device)
        self.acc = torch.tensor(acc, device=self.device)
        self.force = torch.tensor(forces, device=self.device)
        self.grav_acc = torch.tensor([0., -19.8], device=self.device)
        self.mass = torch.full((1, self.num_of_particles), 1.0, device=self.device)
        self.drag = 0.02
        self.air_drag = 0.99
        self.pi = torch.full((1, self.num_of_particles), torch.pi, device=self.device)
        self.boundary_bounce_loss = 0.95
        self.particle_particle_bounce_loss = 0.999
        self.live = torch.full((1, self.num_of_particles), 100, device=self.device)
        self.die = 0
        self.t = temperature
        # self.k_b = 1.380649e-23 # real
        self.k_b = torch.full((1, self.num_of_particles), 1.0, device=self.device)  # for simulation - fake
        self.n_a = 6.0221e+23  # avogardo number
        self.molar_mass = 44.097  # propan  molar mass g/mol
        # self.n = self.mass / self.molar_mass
        self.gas_const = self.k_b * self.n_a
        self.xyz_to_srgb = torch.tensor([
            [3.24100323297636050, -0.96922425220251640, 0.05563941985197549],
            [-1.53739896948878640, 1.87592998369517530, -0.20401120612391013],
            [-0.49861588199636320, 0.04155422634008475, 1.05714897718753330]
        ], device=self.device)

    def get_position(self):
        self.new_pos = self.pos + self.vel * self.dt + 0.5 * self.acc * (self.dt ** 2)
        self.getBrownianMotion()
        self.pos = self.new_pos

    def get_acceleration(self):
        self.new_acc = self.force / self.mass + self.apply_forces()
        self.acc = self.new_acc

    def get_velocity(self, ):
        self.new_vel = self.vel + 0.5 * (self.acc + self.new_acc) * self.dt
        self.new_vel *= self.air_drag

    def apply_forces(self):
        drag_force = 0.5 * self.drag * (self.vel ** 2)
        drag_acc = drag_force / self.mass
        return drag_acc

    def boundaryCollision(self, boundary_min, boundary_max):
        for i in range(2):
            crossed_min = self.pos[i] - self.particleRadius[i] < boundary_min
            crossed_max = self.pos[i] + self.particleRadius[i] > boundary_max
            self.pos[i][crossed_min] = boundary_min + self.particleRadius[i][crossed_min]
            self.vel[i][crossed_min] = -self.vel[i][crossed_min] * self.boundary_bounce_loss
            self.pos[i][crossed_max] = boundary_max - self.particleRadius[i][crossed_max]
            self.vel[i][crossed_max] = -self.vel[i][crossed_max] * self.boundary_bounce_loss

    def particleCollision(self, i, ii):
        # dx = -(self.pos[0][:, None] - self.pos[0].ravel()).reshape(*self.pos[0].shape, *self.pos[0].shape)
        # dy = -(self.pos[1][:, None] - self.pos[1].ravel()).reshape(*self.pos[1].shape, *self.pos[1].shape)
        # distance = torch.hypot(dx, dy)
        # particle2particleRadius = (self.particleRadius[0][:, None] + self.particleRadius[0].ravel()).reshape(
        #     *self.particleRadius[0].shape, *self.particleRadius[0].shape)
        #
        # print(particle2particleRadius)
        interaction = self.interaction_matrix[i, ii]
        dx = self.pos[0][i] - self.pos[0][ii]
        dy = self.pos[1][i] - self.pos[1][ii]
        distance = torch.hypot(dx, dy)
        # distance = np.linalg.norm(other_particle.pos - self.pos)
        particle2particleRadius = (self.particleRadius[0][i] + self.particleRadius[0][ii])

        distance_vector = (self.pos[:][:, i] - self.pos[:][:, ii])
        # normal = distance_vector / distance
        if distance != 0:
            normal = distance_vector / distance
        else:
            normal = torch.zeros_like(distance_vector, device=self.device)
        if distance < particle2particleRadius and interaction == 0:
            v1i = self.vel[:][:, i].clone()
            v2i = self.vel[:][:, ii].clone()
            m1 = self.mass[:][:, i]
            m2 = self.mass[:][:, i]
            v1f = ((m1 - m2) * v1i + 2 * m2 * v2i) / (m1 + m2)
            v2f = ((m2 - m1) * v2i + 2 * m1 * v1i) / (m1 + m2)
            self.vel[:][:, i] = v1f
            self.vel[:][:, ii] = v2f
            self.vel *= self.particle_particle_bounce_loss
            self.vel[:][:, ii] *= self.particle_particle_bounce_loss
            self.interaction_matrix[i, ii] = 1
        elif interaction == 1 and distance > particle2particleRadius:
            self.interaction_matrix[i, ii] = 0
        elif interaction == 1 and distance < particle2particleRadius:
            repulsion = torch.multiply(normal, particle2particleRadius - distance)
            self.pos[:][:, i] -= repulsion / self.mass[:][:, i]
            self.pos[:][:, ii] += repulsion / self.mass[:][:, ii]
        else:
            pass

    def temp2vel_rms(self):
        v = np.sqrt(3 * self.k_b * self.t / (math.pi * self.mass))
        vth = np.array([v, v])
        self.vel += vth

    def vel_rms2temp(self):
        tK = (self.vel ** 2 * self.pi) / (2 * self.k_b * self.mass)
        self.t = (2. / 3.) * torch.mean(tK, dim=0)
        # self.t = np.mean((2. / 3.) * (self.num_of_particles / self.n * self.gas_const) * ((1. / 2.) * self.mass * self.vel ** 2))

    def k2rgb(self):
        # Clamping the temperature between infrared and 40000 Kelvin
        infrared = 500

        self.t = torch.clamp(self.t, infrared, 40000)

        red = torch.zeros_like(self.t, device=self.device)
        green = torch.zeros_like(self.t, device=self.device)
        blue = torch.zeros_like(self.t, device=self.device)
        interp_factor = (self.t - infrared) / (1000 - 273)
        between_mask = (self.t >= infrared) & (self.t <= 1000)
        red[between_mask] = interp_factor[between_mask] * 255
        green[between_mask] = interp_factor[between_mask] * 255
        blue[between_mask] = interp_factor[between_mask] * 255
        above_1000_mask = self.t > 1000
        tmp_internal = self.t[above_1000_mask] / 100.0

        red[above_1000_mask] = torch.where(tmp_internal <= 66,
                                           255,
                                           329.698727446 * torch.pow(tmp_internal - 60, -0.1332047592))
        green[above_1000_mask] = torch.where(tmp_internal <= 66,
                                             99.4708025861 * torch.log(tmp_internal) - 161.1195681661,
                                             288.1221695283 * torch.pow(tmp_internal - 60, -0.0755148492))
        blue[above_1000_mask] = torch.where(tmp_internal >= 66,
                                            255,
                                            torch.where(tmp_internal <= 19,
                                                        0,
                                                        138.5177312231 * torch.log(tmp_internal - 10) - 305.0447927307))

        redh = torch.nn.functional.hardtanh(red / 255.0, min_val=0., max_val=1.)
        greenh = torch.nn.functional.hardtanh(green / 255.0, min_val=0., max_val=1.)
        blueh = torch.nn.functional.hardtanh(blue / 255.0, min_val=0., max_val=1.)

        return redh, greenh, blueh, self.opacity[0]

    def getColorfromTemperature(self):
        self.particleColor = self.k2rgb()
        # print(self.particleColor)
        # r, g, b = self.k2rgb()
        # r, g, b = torch.nn.functional.hardtanh(r, min_val=-0., max_val=1.).item(), torch.nn.functional.hardtanh(g, min_val=-0., max_val=1.).item(), torch.nn.functional.hardtanh(b, min_val=-0., max_val=1.).item()  # Convert tensors to Python floats or integers
        # self.particleColor = [r / 255.0, g / 255.0, b / 255.0] + [self.opacity]

    def getBrownianMotion(self):
        # mean_sq_displacement = torch.nn.functional.mse_loss(self.pos, self.new_pos)
        mean_sq_displacement = (self.new_pos - self.pos) ** 2
        # print(mean_sq_displacement)
        self.acc += self.acc * torch.normal(mean=0., std=mean_sq_displacement)
