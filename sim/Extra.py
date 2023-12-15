import pygame
import numpy as np
grav_const = 0.01 #6.67e-11


def label (text , pos , screen):
    font = pygame.font.Font(None, 20)
    label = font.render(text, True, (255, 255, 255))
    screen.blit(label , pos)

def line (p1 , p2 , screen , color , w=1):
    pygame.draw.aaline(screen, color, p1, p2)


def polar_to_cartesian(angle, radius):
    # Convert angle to radians
    angle_rad = np.radians(angle)
    # Calculate Cartesian coordinates
    x = radius * np.cos(angle_rad)
    y = radius * np.sin(angle_rad)
    return [x, y]


def find_circular_orbit_v(central_obj , rad):
    return np.sqrt(grav_const * central_obj.mass / rad) # tangent

def normalise_vector(v):
    magnitude = np.linalg.norm(v)
    return np.array(v) / magnitude

def calculate_distance(v1 , v2):
    return np.linalg.norm(np.array(v2) - np.array(v1))

def calculate_vector(v1 , v2):
    return np.array(v2) - np.array(v1)



def calc_semi_major_axis(v , r , c_obj , G=grav_const):
    """
    In:
        v : current velocity
        r : current orbit radius (not altitude)
        c_obj : centre object
    Out:
        a : semi-major axis 
    """
    M = c_obj.mass
    v = np.linalg.norm(v)

    p1 = -G*M
    p2 = 2*( (v**2)/2 - G*M/r )

    a = p1/p2
    return a


def calc_eccentricity(v , r , c_obj , G=grav_const):
    """
    In:
        v : current velocity
        r : current orbit radius (not altitude)
        c_obj : centre object
    Out:
        e : eccentricity
    """
    M = c_obj.mass
    v = np.linalg.norm(v)

    p1 = 2*( (v**2)/2 - G*M/r ) * (r*v)**2
    p2 = (G*M)**2

    e = np.sqrt(1 + p1/p2)
    return e


def calc_semi_minor_axis(a , e):
    """
    In:
        a : semi-major axis
        e : eccentricity
    Out:
        b : semi-minor axis
    """
    if e > 1:
        e = 1

    b = a * np.sqrt(1 - e**2)
    return b