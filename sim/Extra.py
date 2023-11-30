import pygame
import numpy as np


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