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