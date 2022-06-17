from random import random
import numpy as np
from numpy import linalg as la
import random

from typing import Tuple, List

class Particle():
    """Contains information about an individual particle.
    """
    def __init__(
        self, 
        id: int, 
        pos: np.ndarray, 
        speed: np.ndarray, 
        radius: float, 
        color: Tuple[int, int, int]
    ) -> None:
        """Create a new particle.

        Args:
            id (int): The particle's ID which should be unique.     
            pos (np.ndarray): The particle's initial position. 
                Dimensions: [x, y]
            speed (np.ndarray): The particle's initital speed. This sets the  
                amount of pixels a particle moves between timesteps for each
                dimension separately. 
            radius (float): The particle's radius.
            color (Tuple[int, int, int]): The particle's color. 
        """
        self.id = id
        self.pos = pos
        self.speed = speed
        self.radius = radius
        self.color = color

    def update(self, pos: np.ndarray, speed: np.ndarray) -> None: 
        """Update a particle's current position and speed. 

        Args:
            pos (np.ndarray): The particle's new position.
            speed (np.ndarray): The particle's new speed.
        """
        self.pos = pos
        self.speed = speed


class ParticleSystem():
    def __init__(
        self, 
        num_particles: int, 
        canvas_shape: Tuple[int, int, int], 
        p_min_radius: int, 
        p_max_radius: int, 
        p_colors: List, 
        p_min_start_speed: int, 
        p_max_start_speed: int, 
        p_max_speed: float,
        p_speed_decay_factor: float,
        collision_radius_threshold: float, 
        collision_speed_threshold: float, 
        bounce: float
    ) -> None:
        """Create a new particle system.

        Args:
            num_particles (int): Number of particles in the system.
            canvas_shape (Tuple[int, int, int]): Describes the size of the 
                canvas. First two entries describe it's size in y and x. 
                Particles will be spawned inside canvas dimensions.
            p_min_radius (int): The minimum radius of a particle.
            p_max_radius (int): The maximum radius of a particle.
            p_colors (List): The color palette each particle randomly gets 
                assigned one color from.
            p_min_start_speed (int): The minimum initial speed of a particle.
            p_max_start_speed (int): The maximum initial speed of a particle.
            p_max_speed (float): The highest possible speed a particle can 
                have per dimension. 
            p_speed_decay_factor (float): A constant factor that each 
                particle's speed gets multiplied with after each timestep. A 
                value smaller than one slows particles and vice versa.
            collision_radius_threshold (float): Describes how close to a 
                particle a collider needs to be to trigger a collision with it.
            collision_speed_threshold (float): The minimum speed a collider has
                to travel with. Slower moving colliders will not be considered 
                for collision calculations. 
            bounce (float): A bounce factor to make collisions more pronounced.
        """
        self.particles = self.create_particles(
            num_particles, 
            canvas_shape, 
            p_min_radius, 
            p_max_radius, 
            p_colors, 
            p_min_start_speed, 
            p_max_start_speed, 
            p_max_speed
        )
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
        

    def create_particles(
        self, 
        num_particles: int, 
        canvas_shape: Tuple[int, int, int], 
        min_radius: int, 
        max_radius: int, 
        colors: List, 
        min_start_speed: int, 
        max_start_speed: int, 
        max_speed: float
    ) -> List[Particle]:
        """Creates a list of all particles that live in the particle system.

        Args:
            num_particles (int): See constructor description.
            canvas_shape (Tuple[int, int, int]): See constructor description.
            min_radius (int): See constructor description.
            max_radius (int): See constructor description.
            colors (List): See constructor description.
            min_start_speed (int): See constructor description.
            max_start_speed (int): See constructor description.
            max_speed (float): See constructor description.

        Returns:
            List[Particle]: A list of all particles in the particle system.
        """
        particles = []
        for particle in range(0, num_particles):
            color = random.choice(colors)
            radius = random.randrange(min_radius, max_radius)
            speed = random.randrange(min_start_speed, max_start_speed)
            speed = min(speed, max_speed)
            possible_directions = [[1, 1], [1, -1], [-1, 1], [-1, -1]]
            direction = np.array(random.choice(possible_directions))
            start_speed = direction * speed 
            start_pos = np.array([
                random.randrange(max_radius, canvas_shape[1] - max_radius), 
                random.randrange(max_radius, canvas_shape[0] - max_radius)
            ])
            particles.append(
                Particle(particle, start_pos, start_speed, radius, color)
            )
        return particles


    def update(
        self, 
        collider_pos=np.array([0,0]), 
        collider_speed=np.array([0,0]), 
        use_collider=False
    ) -> None:
        """Update all particles within the particle system.

        Args:
            collider_pos (np.ndarray, optional): The position of the collider. 
                Defaults to np.array([0,0]).
            collider_speed (np.ndarray, optional): The speed of the collider. 
                Defaults to np.array([0,0]).
            use_collider (bool, optional): Indicates if collisions with 
                colliders are calculated. Defaults to False.
        """
        for particle in self.particles:
            old_pos = particle.pos
            collider_distance = la.norm(particle.pos - collider_pos) 
            collider_distance -= particle.radius
            if (
                use_collider and 
                collider_distance < self.collision_radius_threshold
            ):
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

            particle.update(
                (old_pos + new_speed), 
                new_speed * np.full(2, self.p_speed_decay_factor)
            )

    # TODO: change behaviour at canvas borders
    # currently: 
    #   If a particles' next position is detected to be outside the canvas, the 
    #   respective speed dimension is negated.
    #   The updated next position is then derived from the current one and the 
    #   updated speed.
    #   --> Particles can never fully reach the canvas border before turning 
    #       around
    # goal: 
    #   Simulate correct reflection.
    #   However, next position could be again outside canvas in regards to the 
    #   other dimension, needs to be accounted for.
    #   --> Rejection sample new positions/ directions until valid 

    def get_corrected_speed_direction(self, particle: Particle) -> np.ndarray:
        """Correct speed direction of a particle at canvas borders. 

        Args:
            particle (Particle): A particle.

        Returns:
            np.ndarray: The new speed of the particle. If a particle would hit 
            a canvas border in the next timestep, its direction gets mirrored 
            at that border.
        """
        corrected_speed = particle.speed
        if not self.within_x(particle):
            corrected_speed *= np.array([-1, 1])
        if not self.within_y(particle):
            corrected_speed *= np.array([1, -1])
        return corrected_speed

    def within_x(self, particle: Particle) -> bool:
        """Test if a particle will hit a border in the x dimension.

        Args:
            particle (Particle): A particle.

        Returns:
            bool: True if the particle stays within the canvas, False 
            otherwise.
        """
        within_max = particle.pos[0] + particle.radius < self.canvas_shape[1]
        within_min = particle.pos[0] - particle.radius > 0
        return within_max and within_min

    def within_y(self, particle: Particle) -> bool:
        """Test if a particle will hit a border in the y dimension.

        Args:
            particle (Particle): A particle.

        Returns:
            bool: True if the particle stays within the canvas, False 
            otherwise.
        """
        within_max = particle.pos[1] + particle.radius < self.canvas_shape[0]
        within_min = particle.pos[1] - particle.radius > 0
        return within_max and within_min 

    def get_limited_speed(self, speed: np.ndarray) -> np.ndarray:
        """Cap speed at the maximum particle speed of the particle system.

        Args:
            speed (np.ndarray): Speed of a particle.

        Returns:
            np.ndarray: The minimum of the provided speed and the maximum 
            particle speed.
        """
        limited_speed = np.minimum(np.abs(speed), np.full(2, self.p_max_speed))
        limited_speed *= np.sign(speed)
        return limited_speed

