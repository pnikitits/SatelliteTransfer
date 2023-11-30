import pygame
from pygame import gfxdraw
from Planet import planet
import numpy as np
from Extra import *
from environment import BaseEnvironment

WIDTH, HEIGHT = 1000, 800
EARTH_RADIUS = 100
SATELLITE_RADIUS = 7
WHITE = (255, 255, 255)
TIME_STEP = 30

ep_count = "None Found"

VISUALISE = False # add a visualise toggle bool
# maybe only visualise for ep = 1 , 500 , 3000 ?
REACHED_DIST = 10

ACTION_1_is_on = False






class SatelliteEnvironment(BaseEnvironment):
    def __init__(self):
        self.name = "Satellite Simulator"
        self.min_DIST_reached = 10000 # For logging the "loss" instead of the sum of rewards
        
        

    def env_init(self , env_info={}):
        self.min_DIST_reached = 10000 

        if VISUALISE:
            # --- Pygame initialisation --- #
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Satellite Simulator")
            self.clock = pygame.time.Clock()
            # --- Pygame initialisation --- #


        # --- Objects creation --- #
        self.earth = planet(mass=1000 ,
                    name="Earth" ,
                    position=np.array([WIDTH // 2, HEIGHT // 2]))
        self.satellite_1 = planet(mass=1,
                            name="Satellite_1",
                            position=np.array([self.earth.position[0] + EARTH_RADIUS + 60, self.earth.position[1]]))
        self.satellite_1.set_circular_orbit_velocity(central_obj=self.earth, orbit_radius=EARTH_RADIUS + 60)


        sat_2_init_pos = np.array(polar_to_cartesian(-90 , EARTH_RADIUS+140))
        sat_2_init_pos += self.earth.position

        self.satellite_2 = planet(mass=1,
                            name="Satellite_2",
                            position=sat_2_init_pos)
        self.satellite_2.set_circular_orbit_velocity(central_obj=self.earth, orbit_radius=EARTH_RADIUS + 140)
        
        self.pl_array = [self.earth]
        # --- Objects creation --- #

        # --- Other values --- #
        self.sat_1_fuel = 100
        self.initial_distance = np.linalg.norm(self.satellite_1.position - self.satellite_2.position)
        

        observation = self.env_observe_state()
        self.last_observation = observation
        return observation


    

    def values_update(self):
        

        self.satellite_1.update_velocity(self.pl_array , dt=TIME_STEP)
        self.satellite_1.update_pos(dt=TIME_STEP)
        self.satellite_2.update_velocity(self.pl_array , dt=TIME_STEP)
        self.satellite_2.update_pos(dt=TIME_STEP)

        inst_dist = np.linalg.norm(self.satellite_1.position - self.satellite_2.position)
        if inst_dist < self.min_DIST_reached:
            self.min_DIST_reached = inst_dist

    def visual_update(self):
        if VISUALISE == False:
            return
        
        global ACTION_1_is_on
        

        screen = self.screen
        earth = self.earth
        satellite_1 = self.satellite_1
        satellite_2 = self.satellite_2
        screen.fill((3, 9, 41))

        bg = pygame.image.load("img2.jpg")
        bg = pygame.transform.scale(bg, (1000, 800))
        screen.blit(bg, (0, 0))

        # orbits
        #pygame.draw.circle(screen, (135, 135, 135), earth.position, EARTH_RADIUS + 60, 1)
        gfxdraw.aacircle(screen , earth.position[0] , earth.position[1] , EARTH_RADIUS + 60 , (135, 135, 135))
        #pygame.draw.circle(screen, (135, 135, 135), earth.position, EARTH_RADIUS + 140, 1)
        gfxdraw.aacircle(screen , earth.position[0] , earth.position[1] , EARTH_RADIUS + 140 , (135, 135, 135))

        # earth
        pygame.draw.circle(screen, (4, 113, 135), earth.position, EARTH_RADIUS)  # Fill
        #pygame.draw.circle(screen, WHITE, earth.position, EARTH_RADIUS, 1)       # Border
        gfxdraw.aacircle(screen , earth.position[0] , earth.position[1] , EARTH_RADIUS , WHITE)

        # satellite
        line(satellite_1.position , satellite_2.position , screen , (51, 77, 47))
        pygame.draw.circle(screen, WHITE, (int(satellite_1.position[0]),
                                           int(satellite_1.position[1])),
                                           SATELLITE_RADIUS)
        line(satellite_1.position , satellite_1.position + satellite_1.normalise_vector(satellite_1.velocity)*60 , screen , WHITE)
        
        # Show boost direction
        if ACTION_1_is_on:
            #print("action 1")
            line(satellite_1.position , satellite_1.position - satellite_1.get_tangent_vec(self.earth)*40 , screen , (196, 116, 10) , w=4)
        


        pygame.draw.circle(screen, WHITE, (int(satellite_2.position[0]),
                                           int(satellite_2.position[1])),
                                           SATELLITE_RADIUS)
        line(satellite_2.position , satellite_2.position + satellite_2.normalise_vector(satellite_2.velocity)*60 , screen , WHITE)
        

        # Display satellite velocity and altitude
        label(f"Satellite 1" , (WIDTH - 300, 20) , screen)
        label(f"Velocity: {round(np.linalg.norm(satellite_1.velocity), 2)}" , (WIDTH - 300, 60) , screen)
        label(f"Altitude: {round(np.linalg.norm(satellite_1.position - earth.position) - EARTH_RADIUS, 2)}" , (WIDTH - 300, 100) , screen )
        label(f"Satellite 2" , (WIDTH - 170, 20) , screen)
        label(f"Velocity: {round(np.linalg.norm(satellite_2.velocity), 2)}" , (WIDTH - 170, 60) , screen)
        label(f"Altitude: {round(np.linalg.norm(satellite_2.position - earth.position) - EARTH_RADIUS, 2)}" , (WIDTH - 170, 100) , screen )
        label(f"Distance: {round(np.linalg.norm(satellite_1.position - satellite_2.position) , 2)}" , (WIDTH - 170, 140) , screen )
        label("Orbit 1" , (WIDTH - 365, HEIGHT//2 + 80) , screen)
        label("Orbit 2" , (WIDTH - 290, HEIGHT//2 + 110) , screen)
        
        label(ep_count , (20, 20) , screen)
        label(f"Fuel: {self.sat_1_fuel}" , (WIDTH - 300 , 140) , screen)
            
        pygame.display.flip()
        self.clock.tick(60)


    def env_observe_state(self):
        # sat_1 altitude
        sat_1_alt = np.linalg.norm(self.satellite_1.position - self.earth.position) - EARTH_RADIUS
        # dist(sat_1 , sat_2)
        dist = np.linalg.norm(self.satellite_1.position - self.satellite_2.position)
        # sat_1 fuel left
        fuel = self.sat_1_fuel

        #altitude_diff = sat_1_alt - np.linalg.norm(self.satellite_2.position - self.earth.position) - EARTH_RADIUS
        #radial_diff = self.satellite_1.get_angle_in_orbit(self.earth) - self.satellite_2.get_angle_in_orbit(self.earth)
        #print("OBSERVING :" , (sat_1_alt , dist , fuel))
        return (sat_1_alt , dist , fuel , self.min_DIST_reached)


    def perform_action(self , a):
        #print("ACTION :" , a)

        global ACTION_1_is_on
        
        # Observe current state
        current_state = self.env_observe_state()

        # Perform action
        if a == 0:
            self.sat_1_fuel -= 1
            self.satellite_1.change_tangent_velocity(self.earth , 0.01)
            ACTION_1_is_on = True
        elif a == 4:
            # 4: set velocity to stay in orbit
            self.satellite_1.set_circular_orbit_velocity(self.earth , self.satellite_1.calculate_distance(self.satellite_1.position , self.earth.position))
        else:
            ACTION_1_is_on = False
            

        # Observe new state
        next_state = self.env_observe_state()

        # Calculate reward
        reward = self.calculate_reward(current_state, a, next_state)

        is_terminal = self.is_terminal(next_state)
        return (reward, next_state, is_terminal)
        
        
    def define_possible_actions(self):
        # Actions: 0: accelerate, 1: wait
        return [0,1]

    def is_terminal(self , state):
        sat_1_alt , dist , fuel , _ = state

        if sat_1_alt < 0: # Satellite has crashed on Earth
            return True
        elif dist < REACHED_DIST: # Satellite has reached objective
            return True
        elif fuel <= 0: # Satellite has no more fuel
            return True
        elif dist > 600: # Satellite goes too far
            return True
        
        return False 
    

    def calculate_reward(self , state , action , next_state):
        sat_1_alt , dist , fuel , _ = state
        next_sat_1_alt , next_dist , _ , _ = next_state
        reward = 0

        if action == 0: # Using fuel
            reward -= 1

        if dist > REACHED_DIST and next_dist <= REACHED_DIST: # Reaching objective
            reward += 150

        if sat_1_alt > 0 and next_sat_1_alt <= 0: # Crashing on Earth
            reward -= 100

        if fuel <= 0 and dist > REACHED_DIST: # Fail to reach objective
            reward -= 100

        return reward

    

    def env_start(self):
        reward = 0.0
        
        is_terminal = False

        self.values_update()
        self.visual_update()

        observation = self.env_init()
                
        self.reward_obs_term = (reward, observation, is_terminal)
        
        # return first state observation from the environment
        return self.reward_obs_term


    def env_step(self, action):
        # Take a step in the environment based on the given action
        # Perform the action in your environment
        self.perform_action(action)

        

        self.values_update()
        self.visual_update()

        # Observe the new state
        next_state = self.env_observe_state()

        # Check if the episode is terminal
        is_terminal = self.is_terminal(next_state)

        # Calculate the reward for the current state, action, and next state
        reward = self.calculate_reward(self.last_observation, action, next_state)

        # Update the last observation
        self.last_observation = next_state

        # Return the tuple (reward, next_state, is_terminal)
        return (reward, next_state, is_terminal)

    def env_end(self):
        # End the current episode
        pass

    def env_cleanup(self):
        # Clean up the environment
        self.env_init()

    def get_min_dist(self):
        #print("min_dist in ep :" , self.min_DIST_reached)
        return self.min_DIST_reached
        

    def pass_count(self , message):
        global ep_count
        ep_count = message
        
