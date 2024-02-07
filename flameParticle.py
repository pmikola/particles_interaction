import math
import time

import numpy as np
import torch
from torch import jit


class flameParticle(object):
    def __init__(self, pos, vel, acc, force, radius, temperature, num_particles, device):
        self.particleColor = None
        self.id = None
        self.device = device
        self.new_pos = torch.rand(2, device=self.device)
        self.new_vel = torch.rand(2, device=self.device) * 2 - 1
        self.new_acc = torch.tensor([0., 0.], device=self.device)
        self.num_of_particles = num_particles
        self.interaction_matrix = torch.zeros(self.num_of_particles, device=self.device)
        self.time_interval = 1e-6
        self.dt = 0
        self.opacity = 1.
        self.particleRadius = radius
        self.pos = torch.tensor([pos[0], pos[1]], device=self.device)
        self.vel = torch.tensor([vel[0], vel[1]], device=self.device)
        self.acc = torch.tensor([acc[0], acc[1]], device=self.device)
        self.force = torch.tensor([force[0], force[1]], device=self.device)
        self.grav_acc = torch.tensor([0., -1.], device=self.device)
        self.mass = 1.
        self.drag = 0.1
        self.air_drag = 0.99
        self.boundary_bounce_loss = 0.95
        self.particle_particle_bounce_loss = 0.95
        self.live = 100
        self.t = temperature
        # self.k_b = 1.380649e-23 # real
        self.k_b = 1.  # for simulation - fake
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
        for i in range(2):  # Loop over x and y coordinates
            if self.pos[i] - self.particleRadius < boundary_min:
                self.pos[i] = boundary_min + self.particleRadius
                self.vel[i] = -self.vel[i] * self.boundary_bounce_loss

            if self.pos[i] + self.particleRadius > boundary_max:
                self.pos[i] = boundary_max - self.particleRadius
                self.vel[i] = -self.vel[i] * self.boundary_bounce_loss


    def particleCollision(self, other_particle):
        dx = self.pos[0] - other_particle.pos[0]
        dy = self.pos[1] - other_particle.pos[1]
        distance = torch.hypot(dx, dy)
        # distance = np.linalg.norm(other_particle.pos - self.pos)
        particle2particleRadius = (self.particleRadius + other_particle.particleRadius)
        distance_vector = (other_particle.pos - self.pos)
        # normal = distance_vector / distance
        if distance != 0:
            normal = distance_vector / distance
        else:
            normal = torch.zeros_like(distance_vector, device=self.device)

        if distance < particle2particleRadius and self.interaction_matrix[other_particle.id] == 0:
            # tangent = math.atan2(dy, dx)
            # angle = 0.5 * math.pi + tangent
            # angle = 2 * tangent - angle
            (self.vel, other_particle.vel) = (other_particle.vel, self.vel)  # assuming masses are the same
            self.vel *= self.particle_particle_bounce_loss
            other_particle.vel *= self.particle_particle_bounce_loss
            # self.pos[0] += math.sin(angle)
            # self.pos[1] -= math.cos(angle)
            # other_particle.pos[0] -= math.sin(angle)
            # other_particle.pos[1] += math.cos(angle)
            self.interaction_matrix[other_particle.id] = 1
            return other_particle
        elif self.interaction_matrix[other_particle.id] == 1 and distance > particle2particleRadius:
            self.interaction_matrix[other_particle.id] = 0
            return other_particle
        elif self.interaction_matrix[other_particle.id] == 1 and distance < particle2particleRadius:
            repulsion = torch.multiply(normal, particle2particleRadius - distance)
            self.pos -= repulsion / self.mass
            other_particle.pos += repulsion / other_particle.mass
            return other_particle
        else:
            pass

    # def temp2vel_rms(self):
    #     v = np.sqrt(3 * self.k_b * self.t / (math.pi * self.mass))
    #     vth = np.array([v, v])
    #     self.vel = vth
    def vel_rms2temp(self):
        self.t = (2. / 3.) * torch.mean((self.vel ** 2 * torch.pi) / (2 * self.k_b * self.mass))
        # self.t = np.mean((2. / 3.) * (self.num_of_particles / self.n * self.gas_const) * ((1. / 2.) * self.mass * self.vel ** 2))
        # print(self.t)

    def k2rgb(self):
        self.t = torch.clamp(self.t, 1000, 40000)
        tmp_internal = self.t / 100.0

        red = torch.where(tmp_internal <= 66, 255, 329.698727446 * torch.pow(tmp_internal - 60, -0.1332047592))
        green = torch.where(tmp_internal <= 66,
                            99.4708025861 * torch.log(tmp_internal) - 161.1195681661,
                            288.1221695283 * torch.pow(tmp_internal - 60, -0.0755148492))

        blue = torch.where(tmp_internal >= 66, 255,
                           torch.where(tmp_internal <= 19, 0,
                                       138.5177312231 * torch.log(tmp_internal - 10) - 305.0447927307))

        redh = torch.nn.functional.hardtanh(red / 255.0, min_val=0., max_val=1.)
        greenh = torch.nn.functional.hardtanh(green / 255.0, min_val=0., max_val=1.)
        blueh = torch.nn.functional.hardtanh(blue / 255.0, min_val=0., max_val=1.)
        return redh, greenh, blueh


    def getColorfromTemperature(self):

        self.particleColor = list(map(lambda div: div.item(), self.k2rgb())) + [self.opacity]
        # print(self.particleColor)
        # r, g, b = self.k2rgb()
        # r, g, b = torch.nn.functional.hardtanh(r, min_val=-0., max_val=1.).item(), torch.nn.functional.hardtanh(g, min_val=-0., max_val=1.).item(), torch.nn.functional.hardtanh(b, min_val=-0., max_val=1.).item()  # Convert tensors to Python floats or integers
        # self.particleColor = [r / 255.0, g / 255.0, b / 255.0] + [self.opacity]
