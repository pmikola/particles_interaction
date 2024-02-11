import torch


class Quadtree:
    def __init__(self, boundary, particle_capacity):
        self.boundary = boundary
        self.capacity = particle_capacity
        self.points = torch.empty((0, 2))
        self.divided = False

    def insert(self, x, y):
        if not self.boundary.contains(x, y):
            return False
        if len(self.points) < self.capacity:
            self.points = torch.cat((self.points, torch.tensor([[x, y]])))
            return True
        else:

            if not self.divided:
                self.subdivide()
                # Move existing points into child quadrants
                for point in self.points:
                    self.northeast.insert(point[0], point[1])
                    self.northwest.insert(point[0], point[1])
                    self.southeast.insert(point[0], point[1])
                    self.southwest.insert(point[0], point[1])
                self.points = torch.empty((0, 2))  # Reset points for current quadrant
            # Insert the new point into appropriate child quadrant
            if self.northeast.insert(x, y) or self.northwest.insert(x, y) or \
                    self.southeast.insert(x, y) or self.southwest.insert(x, y):
                return True
        return False

    def subdivide(self):
        # New child quadtree
        x = self.boundary.x
        y = self.boundary.y
        w = self.boundary.w
        h = self.boundary.h

        ne = Box(x + w / 2, y - h / 2, w / 2, h / 2)
        nw = Box(x - w / 2, y - h / 2, w / 2, h / 2)
        se = Box(x + w / 2, y + h / 2, w / 2, h / 2)
        sw = Box(x - w / 2, y + h / 2, w / 2, h / 2)

        self.northeast = Quadtree(ne, self.capacity)
        self.northwest = Quadtree(nw, self.capacity)
        self.southeast = Quadtree(se, self.capacity)
        self.southwest = Quadtree(sw, self.capacity)
        self.divided = True

    def query(self, pbox):
        # recursive algorithm of founding particles
        found = torch.empty((0, 2))

        if not self.boundary.intersects(pbox):
            return found
        elif self.points.size(0) != found.size(0):
            if pbox.contains(self.points[0][0], self.points[0][1]):
                found = torch.cat((found, torch.tensor([self.points[0][0], self.points[0][1]])))

        if self.divided:
            found = torch.cat((found, self.northwest.query(pbox)))
            found = torch.cat((found, self.northeast.query(pbox)))
            found = torch.cat((found, self.southwest.query(pbox)))
            found = torch.cat((found, self.southeast.query(pbox)))
        return found


class Box:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def intersects(self, pbox):
        x_1 = (pbox.x - pbox.w / 2. >= self.x + self.w / 2)
        x_2 = (pbox.x + pbox.w / 2. < self.x - self.w / 2)
        y_1 = (pbox.y - pbox.h / 2. >= self.y + self.h / 2)
        y_2 = (pbox.y + pbox.h / 2. < self.y - self.h / 2)
        inter = x_1 | x_2 | y_1 | y_2

        return inter

    def contains(self, x, y):
        in_x = self.x - self.w / 2. <= x <= self.x + self.w / 2
        in_y = self.y - self.h / 2. <= y <= self.y + self.h / 2

        return in_x and in_y
