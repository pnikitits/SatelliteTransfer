import numpy as np

grav_const = 0.01 #6.67e-11

class planet:
    def __init__(self , mass , name ,
                 position=np.array([0,0]) , 
                 velocity=np.array([0,0])):
        
        self.mass = mass
        self.position = position
        self.name = name
        self.velocity = velocity

    def update_velocity(self, pl_array, dt=1 / 30):
        total_acceleration = np.zeros(2 , dtype=float)
        for pl in pl_array:
            if self.name != pl.name:
                distance = self.calculate_distance(self.position, pl.position)
                vector = self.calculate_vector(pl.position, self.position)
                force_direction = self.normalise_vector(vector)
                force_magnitude = grav_const * self.mass * pl.mass / (distance**2)
                force_vector = -force_direction * force_magnitude
                acceleration = force_vector / self.mass
                total_acceleration += acceleration
        self.velocity += total_acceleration * dt

    def set_circular_orbit_velocity(self, central_obj, orbit_radius):
        # Calculate the velocity for a circular orbit
        tangential_velocity = np.sqrt(grav_const * central_obj.mass / orbit_radius)
        # Set the velocity perpendicular to the vector from the satellite to the central mass (Earth)
        vector_to_earth = central_obj.position - self.position
        perpendicular_velocity = np.array([-vector_to_earth[1], vector_to_earth[0]])
        self.velocity = (tangential_velocity / np.linalg.norm(perpendicular_velocity)) * perpendicular_velocity

    def change_tangent_velocity(self , central_obj , v):
        vector_to_earth = central_obj.position - self.position
        perpendicular_velocity = np.array([-vector_to_earth[1], vector_to_earth[0]])
        perp_direction = self.normalise_vector(perpendicular_velocity)
        self.velocity = self.velocity + perp_direction*v

    def update_pos(self , dt=1/30):
        self.position = self.position + self.velocity*dt

    def calculate_distance(self , v1 , v2):
        return np.linalg.norm(np.array(v2) - np.array(v1))
    
    def calculate_vector(self , v1 , v2):
        return np.array(v2) - np.array(v1)
    
    def normalise_vector(self , v):
        magnitude = np.linalg.norm(v)
        return np.array(v) / magnitude
    
    def get_angle_in_orbit(self , centre_obj):
        angle = np.arctan( (self.position[1] - centre_obj.position[1]) / (self.position[0] - centre_obj.position[0]))
    
    def get_tangent_vec(self , centre_obj):
        vector_to_earth = centre_obj.position - self.position
        perpendicular_velocity = np.array([-vector_to_earth[1], vector_to_earth[0]])
        return self.normalise_vector(perpendicular_velocity)