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