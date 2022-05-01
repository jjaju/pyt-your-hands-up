from random import random
import numpy as np
from numpy import linalg as la
import random

class Particle():
    def __init__(self, id, pos, speed, radius, color):
        self.id = id
        self.pos = pos
        self.speed = speed
        self.radius = radius
        self.color = color

    def update(self, pos, speed):
        self.pos = pos
        self.speed = speed


class ParticleSystem():
    def __init__(self, 
                num_particles, 
                canvas_shape, 
                p_min_radius, 
                p_max_radius, 
                p_colors, 
                p_min_start_speed, 
                p_max_start_speed, 
                p_max_speed,
                p_speed_decay_factor=0.98,
                collision_radius_threshold=10, 
                collision_speed_threshold=1, 
                bounce=2
                ):
        self.particles = self.create_particles(num_particles, canvas_shape, p_min_radius, p_max_radius, p_colors, p_min_start_speed, p_max_start_speed, p_max_speed)
        self.canvas_shape = canvas_shape
        self.collision_radius_threshold = collision_radius_threshold
        self.collision_speed_threshold = collision_speed_threshold
        self.bounce = bounce
        self.p_min_radius = p_min_radius
        self.p_max_radius = p_max_radius
        self.p_colors = p_colors 
        self.p_min_start_speed = p_min_start_speed
        self.p_max_start_speed = p_max_start_speed
        self.p_max_speed = p_max_speed
        self.p_speed_decay_factor = p_speed_decay_factor
        

    def create_particles(self, num_particles, canvas_shape, min_radius, max_radius, colors, min_start_speed, max_start_speed, max_speed):
        particles = []
        for particle in range(0, num_particles):
            color = random.choice(colors)
            radius = random.randrange(min_radius, max_radius)
            speed = min(random.randrange(min_start_speed, max_start_speed), max_speed)
            direction = np.array(random.choice([[1, 1], [1, -1], [-1, 1], [-1, -1]]))
            start_speed = direction * speed 
            particles.append(Particle(particle, np.array([random.randrange(max_radius, canvas_shape[1] - max_radius), 
                                                          random.randrange(max_radius, canvas_shape[0] - max_radius)]),
                                      start_speed, radius, color))
        return particles


    def update(self, collider_pos=np.array([0,0]), collider_speed=np.array([0,0]), use_collider=False):
        
        for particle in self.particles:
            old_pos = particle.pos
            if use_collider and la.norm(particle.pos - collider_pos) - particle.radius < self.collision_radius_threshold:
                if la.norm(collider_speed) > self.collision_speed_threshold: 
                    bounce_speed = collider_speed * self.bounce
                else:
                    bounce_speed = particle.speed

                bounce_speed = self.get_limited_speed(bounce_speed)
                new_pos = particle.pos + bounce_speed
                particle.update(new_pos, bounce_speed)
                new_speed = self.get_corrected_speed_direction(particle)
                
            else:
                new_pos = particle.pos + particle.speed 
                particle.update(new_pos, particle.speed)
                new_speed = self.get_corrected_speed_direction(particle)

            particle.update((old_pos + new_speed), new_speed * np.full(2, self.p_speed_decay_factor))#np.minimum(new_speed * np.full(2, self.p_speed_decay_factor), np.full(2, self.p_max_speed)))

    # TODO: change behaviour at canvas borders
    # currently: If a particles' next position is detected to be outside the canvas, the respective speed dimension is negated.
    #            The updated next position is then derived from the current one and the updated speed.
    #            --> Particles can never fully reach the canvas border before turning around
    # goal: Simulate correct reflection.
    #       However, next position could be again outside canvas in regards to the other dimension, needs to be accounted for.
    #       --> Rejection sample new positions/ directions until valid 

    def get_corrected_speed_direction(self, particle):
        new_speed = particle.speed if self.within_x(particle) else particle.speed * np.array([-1, 1])
        corrected_speed = new_speed if self.within_y(particle) else new_speed * np.array([1, -1])
        return corrected_speed

    def within_x(self, particle):
        return particle.pos[0] + particle.radius  < self.canvas_shape[1] and particle.pos[0] - particle.radius > 0

    def within_y(self, particle):
        return particle.pos[1] + particle.radius < self.canvas_shape[0] and particle.pos[1] - particle.radius > 0

    def get_limited_speed(self, speed):
        limited_speed = np.sign(speed) * np.minimum(np.abs(speed), np.full(2, self.p_max_speed))
        return limited_speed

