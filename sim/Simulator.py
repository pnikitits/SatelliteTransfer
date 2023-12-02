import pygame
from pygame import gfxdraw
from Planet import planet
import numpy as np
import matplotlib.pyplot as plt
from Extra import *
from environment import BaseEnvironment


ep_count = "None Found"
action_1_is_on = False


class SatelliteEnvironment(BaseEnvironment):
    def __init__(self):
        self.name = "Satellite Simulator"

        self.time_step = 30
        self.reached_dist = 0.02
        self.boost_strength = 0.01

        self.visualise = False
        self.width , self.height = 1000 , 800
        self.satellite_radius = 7
        self.earth_radius = 100

        self.orbit_1 = self.earth_radius + 60
        self.orbit_2 = self.earth_radius + 140

        
    def env_init(self , env_info={}):
         
        self.min_dist_reached = 10000 # Minimum distance reached during an episode (for logging the "loss" instead of the sum of rewards)
        
        self.log_sat_1_alt = []
        self.log_sat_2_alt = []
        self.log_time_step = []
        self.current_time_step = 0

        self.sat_1_fuel = 100


        # --- Objects creation --- #

        # Earth
        self.earth = planet(mass=1000 ,
                            name="Earth" ,
                            position=np.array([self.width // 2, self.height // 2]))
        
        # Satellite 1 (OTV)
        self.satellite_1 = planet(mass=1,
                                  name="Satellite_1",
                                  position=np.array([self.earth.position[0] + self.orbit_1, self.earth.position[1]]))
        self.satellite_1.set_circular_orbit_velocity(central_obj=self.earth, orbit_radius=self.orbit_1)

        # Satellite 2
        sat_2_init_pos = np.array(polar_to_cartesian(-90 , self.orbit_2))
        sat_2_init_pos += self.earth.position
        self.satellite_2 = planet(mass=1,
                                  name="Satellite_2",
                                  position=sat_2_init_pos)
        self.satellite_2.set_circular_orbit_velocity(central_obj=self.earth, orbit_radius=self.orbit_2)
        
        self.pl_array = [self.earth] # Objects that generate gravity

        # --- Objects creation --- #


        if self.visualise:
            # --- Pygame initialisation --- #
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption(self.name)
            self.clock = pygame.time.Clock()
            # --- Pygame initialisation --- #

        observation = self.env_observe_state()
        self.last_observation = observation
        return observation


    

    def values_update(self):
        self.update_for_plots()

        # Update velocities and positions
        self.satellite_1.update_velocity(self.pl_array , dt=self.time_step)
        self.satellite_1.update_pos(dt=self.time_step)
        self.satellite_2.update_velocity(self.pl_array , dt=self.time_step)
        self.satellite_2.update_pos(dt=self.time_step)

        # Define the dist as the Euclidean distance
        #inst_dist = np.linalg.norm(self.satellite_1.position - self.satellite_2.position)
        #if inst_dist < self.min_dist_reached:
        #    self.min_dist_reached = inst_dist

        # Define the dist as the difference in tangent velocities
        goal_t_velocity = find_circular_orbit_v(self.earth , self.orbit_2)
        current_t_velocity = np.linalg.norm(self.satellite_1.find_velocity_components(self.earth)[1])
        velocity_diff = abs(goal_t_velocity - current_t_velocity)
        dist = velocity_diff*1000

        # log the min distance reached during the episode
        if dist < self.min_dist_reached:
            self.min_dist_reached = dist

        # Update the reached_dist to smaller values as the learning goes on
        if dist < self.reached_dist:
            self.reached_dist = (self.reached_dist + dist)*0.5
            print(f"update REACHED_DIST {self.reached_dist} -> {dist}")



    def env_observe_state(self):
        # sat_1 fuel left
        fuel = self.sat_1_fuel

        # Altitudes
        sat_1_alt = np.linalg.norm(self.satellite_1.position - self.earth.position) - self.earth_radius
        #sat_2_alt = np.linalg.norm(self.satellite_2.position - self.earth.position) - self.earth_radius
        #altitude_diff = abs(sat_1_alt - sat_2_alt) / 80

        # Eucledian distance
        #dist = np.linalg.norm(self.satellite_1.position - self.satellite_2.position)

        # Angles
        #sat_1_ang = self.satellite_1.get_angle_in_orbit(self.earth)
        #sat_2_ang = self.satellite_2.get_angle_in_orbit(self.earth)
        #angle_diff = abs(sat_1_ang - sat_2_ang) / (np.pi/2)


        # Velocity difference
        goal_t_velocity = find_circular_orbit_v(self.earth , self.orbit_2)
        current_t_velocity = np.linalg.norm(self.satellite_1.find_velocity_components(self.earth)[1])
        velocity_diff = abs(goal_t_velocity - current_t_velocity)
        dist = velocity_diff*1000
        

        return (sat_1_alt , dist , fuel , self.min_dist_reached)


    
    def calculate_reward(self , state , action , next_state):
        sat_1_alt , dist , fuel , _ = state
        next_sat_1_alt , next_dist , _ , _ = next_state
        reward = 0

        # Using fuel
        if action == 0: 
            reward -= 1

        # Reaching objective
        if next_dist < self.reached_dist: 
            print(f"reward given ep {ep_count} dist {next_dist}")
            reward += 3000

        # Crashing on Earth
        if sat_1_alt > 0 and next_sat_1_alt <= 0: 
            reward -= 100

        # Run out of fuel
        if fuel <= 0: 
            reward -= 100

        return reward
    

    def is_terminal(self , state):
        sat_1_alt , dist , fuel , _ = state

        if sat_1_alt < 0: # Satellite has crashed on Earth
            return True
        elif dist < self.reached_dist: # Satellite has reached objective
            #print(f"is terminal REACHED ep {ep_count}")
            return True
        elif fuel <= 0: # Satellite has no more fuel
            return True
        elif dist > 600: # Satellite goes too far
            return True
        elif sat_1_alt < 45 or sat_1_alt > 155: # Satellite peforms non pertinent moves (over constraining ?)
            return True
        
        return False 


    def perform_action(self , a):
        global action_1_is_on
        
        # Observe current state
        current_state = self.env_observe_state()

        # Perform action
        if a == 0:
            self.sat_1_fuel -= 1
            self.satellite_1.change_tangent_velocity(self.earth , self.boost_strength)
            action_1_is_on = True
        elif a == 4:
            # 4: set velocity to stay in orbit 
            self.satellite_1.set_circular_orbit_velocity(self.earth , calculate_distance(self.satellite_1.position , self.earth.position))
        else:
            action_1_is_on = False
            
        # Observe new state
        next_state = self.env_observe_state()

        # Calculate reward
        reward = self.calculate_reward(current_state, a, next_state)

        is_terminal = self.is_terminal(next_state)
        return (reward, next_state, is_terminal)


    def env_start(self):
        reward = 0.0
        is_terminal = False

        self.values_update()
        self.visual_update()

        observation = self.env_init()
             
        # return first state observation from the environment
        return (reward, observation, is_terminal)


    def env_step(self, action):
        # Take a step in the environment based on the given action
        # Perform the action in your environment
        self.perform_action(action)

        self.values_update()
        self.visual_update()

        # Observe the new state
        next_state = self.env_observe_state()

        # Calculate the reward for the current state, action, and next state
        reward = self.calculate_reward(self.last_observation, action, next_state)

        # Check if the episode is terminal
        is_terminal = self.is_terminal(next_state)
        # Update the last observation
        self.last_observation = next_state

        # Return the tuple (reward, next_state, is_terminal)
        return (reward, next_state, is_terminal)


    

    

    def update_for_plots(self):
        # Call this once per time step
        self.log_time_step.append(self.current_time_step)

        #sat_1_alt = np.linalg.norm(self.satellite_1.position - self.earth.position) - EARTH_RADIUS
        #sat_2_alt = np.linalg.norm(self.satellite_2.position - self.earth.position) - EARTH_RADIUS
        sat_1_alt = np.linalg.norm(self.satellite_1.find_velocity_components(self.earth)[1])
        sat_2_alt = np.linalg.norm(self.satellite_2.find_velocity_components(self.earth)[1])

        self.log_sat_1_alt.append(sat_1_alt)
        self.log_sat_2_alt.append(sat_2_alt)

        self.current_time_step += 1

    def plot_alts(self , ep):
        x = self.log_time_step
        y1 = self.log_sat_1_alt
        y2 = self.log_sat_2_alt

        plt.plot(x, y1, label='sat_1 (OTV)')
        plt.plot(x, y2, label='sat_2')

        plt.xlabel('Time Step')
        plt.ylabel('Altitude')
        plt.title(f'Episode {ep}')
        plt.legend()
        plt.show()






    def env_end(self):
        # End the current episode
        pass

    def env_cleanup(self):
        # Clean up the environment
        self.env_init()

    def define_possible_actions(self):
        # Actions: 0: accelerate, 1: wait
        return [0,1]
    
    def get_min_dist(self):
        #if self.min_dist_reached <= 20:
        #    print("min_dist in ep :" , self.min_dist_reached)
        return self.min_dist_reached
    
    def pass_count(self , message):
        global ep_count
        ep_count = message



    def visual_update(self):
        if self.visualise == False:
            return
        
        global action_1_is_on
        white = (255, 255, 255)
        

        screen = self.screen
        earth = self.earth
        satellite_1 = self.satellite_1
        satellite_2 = self.satellite_2
        screen.fill((3, 9, 41))

        bg = pygame.image.load("img2.jpg")
        bg = pygame.transform.scale(bg, (1000, 800))
        screen.blit(bg, (0, 0))

        # orbits
        gfxdraw.aacircle(screen , earth.position[0] , earth.position[1] , self.orbit_1 , (135, 135, 135))
        gfxdraw.aacircle(screen , earth.position[0] , earth.position[1] , self.orbit_2 , (135, 135, 135))

        # earth
        pygame.draw.circle(screen, (4, 113, 135), earth.position, self.earth_radius)  # Fill
        gfxdraw.aacircle(screen , earth.position[0] , earth.position[1] , self.earth_radius , white) # Border

        # satellite
        line(satellite_1.position , satellite_2.position , screen , (51, 77, 47))
        pygame.draw.circle(screen, white, (int(satellite_1.position[0]),
                                                int(satellite_1.position[1])),
                                                self.satellite_radius)
        line(satellite_1.position , satellite_1.position + normalise_vector(satellite_1.velocity)*60 , screen , white)
        
        # Show boost direction
        if action_1_is_on:
            line(satellite_1.position , satellite_1.position - satellite_1.get_tangent_vec(self.earth)*40 , screen , (196, 116, 10) , w=4)
        
        pygame.draw.circle(screen, white, (int(satellite_2.position[0]),
                                                int(satellite_2.position[1])),
                                                self.satellite_radius)
        line(satellite_2.position , satellite_2.position + normalise_vector(satellite_2.velocity)*60 , screen , white)
        

        # Display satellite velocity, altitude, ...
        label(f"Satellite 1" , (self.width - 300, 20) , screen)
        label(f"Velocity: {round(np.linalg.norm(satellite_1.velocity), 2)}" , (self.width - 300, 60) , screen)
        label(f"Altitude: {round(np.linalg.norm(satellite_1.position - earth.position) - self.earth_radius, 2)}" , (self.width - 300, 100) , screen )
        label(f"Satellite 2" , (self.width - 170, 20) , screen)
        label(f"Velocity: {round(np.linalg.norm(satellite_2.velocity), 2)}" , (self.width - 170, 60) , screen)
        label(f"Altitude: {round(np.linalg.norm(satellite_2.position - earth.position) - self.earth_radius, 2)}" , (self.width - 170, 100) , screen )
        label(f"Distance: {round(np.linalg.norm(satellite_1.position - satellite_2.position) , 2)}" , (self.width - 170, 180) , screen )
        label("Orbit 1" , (self.width - 365, self.height//2 + 80) , screen)
        label("Orbit 2" , (self.width - 290, self.height//2 + 110) , screen)
        label(ep_count , (20, 20) , screen)
        label(f"Fuel: {self.sat_1_fuel}" , (self.width - 300 , 180) , screen)
        label(f"Angle: {round(self.satellite_1.get_angle_in_orbit(self.earth , deg=True),2)}" , (self.width - 300, 140) , screen)
        label(f"Angle: {round(self.satellite_2.get_angle_in_orbit(self.earth , deg=True),2)}" , (self.width - 170, 140) , screen)
            
        pygame.display.flip()
        self.clock.tick(60)