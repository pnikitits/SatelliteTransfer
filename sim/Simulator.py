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
        self.reached_dist = 10 #tighten this
        self.boost_strength = 0.001

        self.steps_in_reward = 0
        self.max_steps_in_reward = 200
        self.success = False

        self.visualise = False
        self.width , self.height = 1000 , 800
        self.satellite_radius = 7
        self.earth_radius = 100

        self.orbit_1 = self.earth_radius + 60
        self.orbit_2 = self.earth_radius + 140

        
    def env_init(self , env_info={}):
         
        self.min_dist_reached = 10000 # Minimum distance reached during an episode (for logging the "loss" instead of the sum of rewards)
        

        # New
        self.log_alt_sat_1 = [] # [float]
        self.log_alt_sat_2 = [] # [float]
        self.log_alt_boost = [] # [float] = alt_sat_1
        self.log_alt_time  = [] # [int]

        self.log_traj_sat_1 = [] # [[x,y] , ...]
        self.log_traj_sat_2 = [] # [[x,y] , ...]
        self.log_traj_boost = [] # [[x,y] , ...]

        self.log_distance = []


        self.current_time_step = 0

        self.sat_1_fuel = 1000


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
        self.satellite_1.position_where_thrust = []
        self.satellite_1.position_where_thrust2 = []
        
        # Satellite 2
        sat_2_init_pos = np.array(polar_to_cartesian(-60 , self.orbit_2))
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
        dist = np.linalg.norm(self.satellite_1.position - self.satellite_2.position)

        # Define the dist as the difference in tangent velocities
        #goal_t_velocity = find_circular_orbit_v(self.earth , self.orbit_2)
        #current_t_velocity = np.linalg.norm(self.satellite_1.find_velocity_components(self.earth)[1])
        #velocity_diff = abs(goal_t_velocity - current_t_velocity)
        #dist = velocity_diff*1000

        # Altitudes
        #sat_1_alt = np.linalg.norm(self.satellite_1.position - self.earth.position) - self.earth_radius
        #sat_2_alt = self.orbit_2 - self.earth_radius#np.linalg.norm(self.satellite_2.position - self.earth.position) - self.earth_radius
        #dist = abs(sat_1_alt - sat_2_alt)

        # log the min distance reached during the episode
        if dist < self.min_dist_reached:
            self.min_dist_reached = dist

        # Update the reached_dist to smaller values as the learning goes on
        #if dist < self.reached_dist:
        #    self.reached_dist = (self.reached_dist + dist)*0.5
        #    print(f"update REACHED_DIST {self.reached_dist} -> {dist}")



    def env_observe_state(self):

        # Altitudes
        sat_1_alt = np.linalg.norm(self.satellite_1.position - self.earth.position) - self.earth_radius
        sat_2_alt = self.orbit_2 - self.earth_radius#np.linalg.norm(self.satellite_2.position - self.earth.position) - self.earth_radius
        alt_diff = sat_1_alt - sat_2_alt

        # Eucledian distance
        # dist = np.linalg.norm(self.satellite_1.position - self.satellite_2.position)

        # Angles
        sat_1_ang = self.satellite_1.get_angle_in_orbit(self.earth)
        sat_2_ang = self.satellite_2.get_angle_in_orbit(self.earth)
        angle_diff = abs(sat_1_ang - sat_2_ang) / (np.pi/2)


        # Velocity difference
        goal_t_velocity = np.linalg.norm(self.satellite_2.find_velocity_components(self.earth)[0])
        current_t_velocity = np.linalg.norm(self.satellite_1.find_velocity_components(self.earth)[0])
        velocity_diff = (goal_t_velocity - current_t_velocity)*100
        #dist = velocity_diff*1000
        
        # print(f"alt_diff: {alt_diff} , velocity_diff: {velocity_diff}")
        return (alt_diff , velocity_diff)


    
    def calculate_reward(self , action):

        global steps_in_reward
        
        # Calculate metrics needed for reward
        dist = np.linalg.norm(self.satellite_1.position - self.satellite_2.position)
        sat_1_alt = np.linalg.norm(self.satellite_1.position - self.earth.position) - self.earth_radius
        sat_2_alt = self.orbit_2 - self.earth_radius#np.linalg.norm(self.satellite_2.position - self.earth.position) - self.earth_radius
        alt_diff = abs(sat_1_alt - sat_2_alt)

        reward = 0
        if alt_diff < 8:
            reward = 1
            steps_in_reward += 1
            # print('steps_in_reward' , steps_in_reward)
        else:
            steps_in_reward = 0

        # if steps_in_reward > self.max_steps_in_reward/2:
        #     reward += 1
        #     print('half reward')

        # if steps_in_reward > self.max_steps_in_reward:
        #     reward += 1
        #     print('full reward')
        #     self.success = True

        if sat_1_alt > 700: # Satellite goes too far
            reward -= 100
            print('too far reward')
        
        # Using fuel
        if action == 0:
            reward -= 0.5

        # Crashing on Earth
        if sat_1_alt < 0: 
            reward -= 100
            print('crashed reward')

        return reward
    

    def is_terminal(self):
        # # Calculate metrics needed for terminal
        sat_1_alt = np.linalg.norm(self.satellite_1.position - self.earth.position) - self.earth_radius

        if sat_1_alt < 0: # Satellite has crashed on Earth
            print('crashed terminal')
            return True
        elif self.sat_1_fuel <= 0: # Satellite has no more fuel
            return True
        elif sat_1_alt > 700: # Satellite goes too far
            return True
        elif self.success: # Satellite has reached the goal
            return True
        
        return False 


    def perform_action(self , a):
        global action_1_is_on, action_2_is_on

        # Perform action
        if a == 1:
            self.sat_1_fuel -= 1
            self.satellite_1.change_tangent_velocity(self.earth , self.boost_strength)
            action_1_is_on = True
            self.satellite_1.position_where_thrust.append((self.satellite_1.position,2))
        elif a == 2:
            self.sat_1_fuel -= 1
            self.satellite_1.change_tangent_velocity(self.earth , self.boost_strength*5)
            self.satellite_1.position_where_thrust.append((self.satellite_1.position,4))
            action_1_is_on = True
        elif a == 3:
            self.sat_1_fuel -= 1
            self.satellite_1.change_tangent_velocity(self.earth , self.boost_strength*10)
            self.satellite_1.position_where_thrust.append((self.satellite_1.position,6))
            action_1_is_on = True       
        elif a == 4:
            self.sat_1_fuel -= 1
            self.satellite_1.change_tangent_velocity(self.earth , self.boost_strength*20)
            self.satellite_1.position_where_thrust.append((self.satellite_1.position,8))
            action_1_is_on = True       
        # Reverse
        elif a == 5:
            self.sat_1_fuel -= 1
            self.satellite_1.change_tangent_velocity(self.earth , -self.boost_strength)
            action_2_is_on = True
            self.satellite_1.position_where_thrust2.append(self.satellite_1.position)
        # elif a == 4:
        #     # 4: set velocity to stay in orbit 
        #     self.satellite_1.set_circular_orbit_velocity(self.earth , calculate_distance(self.satellite_1.position , self.earth.position))
        else:
            action_1_is_on = False
            action_2_is_on = False


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
        # Applies the velocity change to the satellites
        self.perform_action(action)

        # Update values to to the new state
        # Compute the new position of the satellites
        self.values_update()
        self.visual_update()

        # Observe the new state
        next_state = self.env_observe_state()

        # Calculate the reward for the new state
        reward = self.calculate_reward(action)

        # Check if the episode is terminal
        is_terminal = self.is_terminal()
        # Update the last observation
        self.last_observation = next_state

        # Return the tuple (reward, next_state, is_terminal)
        return (reward, next_state, is_terminal)


    

    

    def update_for_plots(self):
        # Call this once per time step


        # Altitude subplot
        sat_1_alt = np.linalg.norm(self.satellite_1.position - self.earth.position) - self.earth_radius
        sat_2_alt = np.linalg.norm(self.satellite_2.position - self.earth.position) - self.earth_radius
        self.log_alt_sat_1.append(sat_1_alt)
        self.log_alt_sat_2.append(sat_2_alt)
        if action_1_is_on:
            self.log_alt_boost.append(sat_1_alt)
        else:
            self.log_alt_boost.append(np.nan)
        self.log_alt_time.append(self.current_time_step)


        # Trajectory subplot
        sat_1_pos = self.satellite_1.position
        sat_2_pos = self.satellite_2.position
        self.log_traj_sat_1.append(sat_1_pos)
        self.log_traj_sat_2.append(sat_2_pos)
        if action_1_is_on:
            self.log_traj_boost.append(sat_1_pos)
        else:
            self.log_traj_boost.append([np.nan , np.nan])

        # Distance subplot
        self.log_distance.append(np.linalg.norm(self.satellite_1.position - self.satellite_2.position))

        
        # Tangent velocity
        #sat_1_alt = np.linalg.norm(self.satellite_1.find_velocity_components(self.earth)[1])
        #sat_2_alt = np.linalg.norm(self.satellite_2.find_velocity_components(self.earth)[1])

        self.current_time_step += 1

    def plot_alts(self , ep):



        plt.subplot(1, 3, 1)
        plt.plot(*zip(*self.log_traj_sat_1) , linestyle='-', color='b' , label='sat_1 (OTV)')
        plt.plot(*zip(*self.log_traj_sat_2) , linestyle='-', color='g' , label='sat_2')
        plt.plot(*zip(*self.log_traj_boost) , marker='o', color='r' , label='boost')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis('equal')
        plt.gca().invert_yaxis()
        plt.title('Trajectory')
        plt.legend()


        plt.subplot(1, 3, 2)
        x = self.log_alt_time
        y1 = self.log_alt_sat_1
        y2 = self.log_alt_sat_2
        y3 = self.log_alt_boost
        plt.plot(x, y1 , linestyle='-', color='b' , label='sat_1 (OTV)')
        plt.plot(x, y2 , linestyle='-', color='g' , label='sat_2')
        plt.plot(x, y3, marker='o' , color='r' , label='boost')
        plt.xlabel('Time Step')
        plt.ylabel('Altitude')
        plt.title('Altitude')
        plt.legend()


        plt.subplot(1, 3, 3)
        plt.plot(x , self.log_distance)
        plt.xlabel('Time Step')
        plt.ylabel('Distance')
        plt.title('Distance')


        plt.tight_layout()
        plt.suptitle(f'Episode {ep}')
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
        
        # Show boost location
        for x, size in self.satellite_1.position_where_thrust:
            pygame.draw.circle(screen, (255, 0, 0), x, size)
        for x in self.satellite_1.position_where_thrust2:
            pygame.draw.circle(screen, (0, 255, 0), x, 2)

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