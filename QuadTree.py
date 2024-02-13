import torch


class Quadtree:
    def __init__(self, boundary, particle_capacity, device):
        self.boundary = boundary
        self.capacity = particle_capacity
        self.divided = False
        self.AA = None
        self.AB = None
        self.BB = None
        self.BA = None
        self.device = device
        self.particles = torch.empty((0, 2), dtype=torch.float32)

    def subdivide(self):
        x = self.boundary.x
        y = self.boundary.y
        w = self.boundary.w
        h = self.boundary.h

        aa = Box(x, y, w / 2, h / 2)
        ab = Box(x + w / 2, y, w / 2, h / 2)
        ba = Box(x, y + h / 2, w / 2, h / 2)
        bb = Box(x + w / 2, y + h / 2, w / 2, h / 2)

        self.AA = Quadtree(aa, self.capacity, self.device)
        self.AB = Quadtree(ab, self.capacity, self.device)
        self.BA = Quadtree(ba, self.capacity, self.device)
        self.BB = Quadtree(bb, self.capacity, self.device)

        for i in range(len(self.particles)):
            self.AA.insert(self.particles[i][0], self.particles[i][1])
            self.AB.insert(self.particles[i][0], self.particles[i][1])
            self.BA.insert(self.particles[i][0], self.particles[i][1])
            self.BB.insert(self.particles[i][0], self.particles[i][1])

    def insert(self, x, y):
        if not self.boundary.contains(x, y):
            return False
        if len(self.particles) < self.capacity and self.AA is None:
            add_particle = torch.tensor([[x, y]], device=self.device)
            self.particles = torch.cat((self.particles, add_particle))
            return True
        else:
            if self.AA is None:
                self.subdivide()
            if self.AA.insert(x, y):
                return True
            if self.AB.insert(x, y):
                return True
            if self.BA.insert(x, y):
                return True
            if self.BB.insert(x, y):
                return True
            return False

    def query(self, pbox):
        found = torch.empty((0, 2), dtype=torch.float32)

        if not pbox.intersects(self.boundary) is False:  # should be Tr
            return found

        # if self.particles.nelement() != 0:
        for particle in self.particles:
            if pbox.contains(particle[0], particle[1]):
                add_particle = torch.tensor([[particle[0], particle[1]]], device=self.device)
                found = torch.cat((found, add_particle))

        if self.AA is not None:
            found = torch.cat((found, self.AA.query(pbox)))
            found = torch.cat((found, self.AB.query(pbox)))
            found = torch.cat((found, self.BA.query(pbox)))
            found = torch.cat((found, self.BB.query(pbox)))
        return found


class Box:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def intersects(self, pbox):
        # x_1 = (pbox.x - pbox.w / 2. > self.x + self.w / 2)
        # x_2 = (pbox.x + pbox.w / 2. < self.x - self.w / 2)
        # y_1 = (pbox.y - pbox.h / 2. > self.y + self.h / 2)
        # y_2 = (pbox.y + pbox.h / 2. < self.y - self.h / 2)
        # inter = x_1 | x_2 | y_1 | y_2

        return pbox.x > self.x + self.w or pbox.x + pbox.w < self.x - self.w or pbox.y > self.y + self.h or pbox.y + pbox.h < self.y - self.h
        # return inter

    def contains(self, x, y):
        # in_x = self.x - self.w / 2. <= x <= self.x + self.w / 2
        # in_y = self.y - self.h / 2. <= y <= self.y + self.h / 2
        return self.x <= x <= self.x + self.w and self.y <= y <= self.y + self.h
        # return in_x and in_y
