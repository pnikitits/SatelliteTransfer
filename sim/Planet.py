import numpy as np
from Extra import *



class planet:
    def __init__(self , mass , name , radius ,
                 position=np.array([0,0]) , 
                 velocity=np.array([0,0])):
        
        self.mass = mass
        self.position = position
        self.name = name
        self.velocity = velocity
        self.radius = radius


    def update_velocity(self, pl_array, dt=1 / 30):
        total_acceleration = np.zeros(2 , dtype=float)
        for pl in pl_array:
            if self.name != pl.name:
                distance = calculate_distance(self.position, pl.position)
                vector = calculate_vector(pl.position, self.position)
                force_direction = normalise_vector(vector)
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
        perp_direction = normalise_vector(perpendicular_velocity)
        self.velocity = self.velocity + perp_direction*v


    def update_pos(self , dt=1/30):
        self.position = self.position + self.velocity*dt








    """def get_tangent_velocity(self , centre_obj):
        return self.convert_velocity(centre_obj)[0]

    def get_normal_velocity(self, centre_obj):
        return self.convert_velocity(centre_obj)[1]
        
    
    def convert_velocity(self , centre_obj):
        
        #Convert the object's [vx,vy] to [vt,vn]
        
        vx = self.velocity[0]
        vy = self.velocity[1]
        mag = np.sqrt(vx**2 + vy**2)

        dx = self.position[0] - centre_obj.position[0]
        dy = self.position[1] - centre_obj.position[1]
        theta = np.arctan(dy/dx)
        phi = np.arctan(vy/vx) - theta

        vt = mag * np.sin(phi)
        vn = -mag * np.cos(phi)

        return [vt , vn]"""

        








    def get_altitude(self, centre_obj):
        c_to_c_dist = np.linalg.norm(self.position - centre_obj.position)
        altitude = c_to_c_dist - centre_obj.radius
        return altitude




    def get_tangent_vec(self , centre_obj):
        vector_to_earth = centre_obj.position - self.position
        perpendicular_velocity = np.array([-vector_to_earth[1], vector_to_earth[0]])
        return normalise_vector(perpendicular_velocity)

    #def calculate_distance(self , v1 , v2):
    #    return np.linalg.norm(np.array(v2) - np.array(v1))
    
    #def calculate_vector(self , v1 , v2):
    #    return np.array(v2) - np.array(v1)
    
    #def normalise_vector(self , v):
    #    magnitude = np.linalg.norm(v)
    #    return np.array(v) / magnitude
    
    """
    def get_angle_in_orbit(self , centre_obj , deg=False):
        angle = np.arctan( (self.position[1] - centre_obj.position[1]) / (self.position[0] - centre_obj.position[0]))
        if deg:
            return np.degrees(angle)
        return angle

    
    def get_tangent_vec(self , centre_obj):
        vector_to_earth = centre_obj.position - self.position
        perpendicular_velocity = np.array([-vector_to_earth[1], vector_to_earth[0]])
        return normalise_vector(perpendicular_velocity)
    

    def find_velocity_components(self, central_object):
        central_position = central_object.position
        # Step 1: Calculate the relative position vector
        relative_position = self.position - central_position

        # Step 2: Decompose the velocity vector into components
        radial_direction = np.linalg.norm(relative_position)
        radial_unit_vector = relative_position / radial_direction

        radial_velocity = np.dot(self.velocity, radial_unit_vector)
        tangent_velocity = self.velocity - radial_velocity * radial_unit_vector

        return radial_velocity, tangent_velocity
    """