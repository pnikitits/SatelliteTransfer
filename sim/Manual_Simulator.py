import pygame
import sys
from Simulator import SatelliteEnvironment


if __name__ == "__main__":
    sim = SatelliteEnvironment()
    sim.env_init()
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            # Key actions to play with the simulator
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    sim.perform_action(0)
                elif event.key == pygame.K_LEFT:
                    sim.perform_action(4)
                elif event.key == pygame.K_RIGHT:
                    sim.env_init()

        sim.values_update()
        sim.visual_update()


