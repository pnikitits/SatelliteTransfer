import pygame
from pygame import gfxdraw
from Planet import planet
import numpy as np
import matplotlib.pyplot as plt
from Extra import *
from environment import BaseEnvironment


ep_count = "None Found"
action_1_is_on = False
action_2_is_on = False


class SatelliteEnvironment(BaseEnvironment):
    def __init__(self):
        self.name = "Satellite Simulator"

        self.time_step = 40
        self.boost_strength = 0.003

        self.reached_dist = 10
        self.reach_r_velocity = 0.1 # min value observed: 0.0006 !!
        self.reach_dv_tan = 0.01


        self.visualise = False
        self.width , self.height = 1000 , 800

        
        self.satellite_radius = 7
        self.earth_radius = 100

        self.orbit_1 = self.earth_radius + 60
        self.orbit_2 = self.earth_radius + 140

        

        
    def env_init(self , env_info={}):

        self.MIN_RAD_V_IN_REACH = 1000 # As a loss to plot: the minimum velocity in radial direction IN the reach distance
        self.MIN_TAN_DV = 1000
        self.MIN_GG_DIST = 100000
        
        self.action_done = 0

        self.goal_is_done = False

        # Loging
        self.log_alt_sat_1 = [] # [float]
        self.log_alt_sat_2 = [] # [float]
        self.log_alt_boost_pos = [] # [float] = alt_sat_1
        self.log_alt_boost_neg = [] # [float] = alt_sat_1
        self.log_alt_time  = [] # [int]

        self.log_traj_sat_1 = [] # [[x,y] , ...]
        self.log_traj_sat_2 = [] # [[x,y] , ...]
        self.log_traj_boost_pos = [] # [[x,y] , ...]
        self.log_traj_boost_neg = [] # [[x,y] , ...]

        self.log_distance = []

        self.log_radial_v = []
        self.log_tan_v_sat_1 = []
        self.log_tan_v_sat_2 = []


        self.current_time_step = 0

        self.fuel = 100


        # --- Objects creation --- #

        # Earth
        self.earth = planet(mass=1000 ,
                            name="Earth" ,
                            radius=self.earth_radius ,
                            position=np.array([self.width // 2, self.height // 2]))
        
        # Satellite 1 (OTV)
        self.satellite_1 = planet(mass=1,
                                  name="Satellite_1",
                                  radius=self.satellite_radius,
                                  position=np.array([self.earth.position[0] + self.orbit_1, self.earth.position[1]]))
        self.satellite_1.set_circular_orbit_velocity(central_obj=self.earth, orbit_radius=self.orbit_1)

        self.satellite_1.position_where_thrust = []
        self.satellite_1.position_where_thrust2 = []

        # Satellite 2
        sat_2_init_pos = np.array(polar_to_cartesian(-67 , self.orbit_2))
        sat_2_init_pos += self.earth.position
        self.satellite_2 = planet(mass=1,
                                  name="Satellite_2",
                                  radius=self.satellite_radius,
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
        

        #print(self.satellite_1.get_normal_velocity(self.earth))
        #print(f"sat2: vr= {self.satellite_2.get_normal_velocity(self.earth)} | vt= {self.satellite_2.get_tangent_velocity(self.earth)}")

        
        # ------------ For plotting at end of training ------------ #
        dist = np.linalg.norm(self.satellite_1.position - self.satellite_2.position)

        #sat1_r_velocity = abs(self.satellite_1.get_normal_velocity(self.earth))

        #if dist < self.reached_dist:
        #    if sat1_r_velocity < self.MIN_RAD_V_IN_REACH:
        #        self.MIN_RAD_V_IN_REACH = sat1_r_velocity

        #sat_1_tan_v = self.satellite_1.get_tangent_velocity(self.earth)
        #sat_2_tan_v = self.satellite_2.get_tangent_velocity(self.earth)
        #dv_tan = abs(sat_1_tan_v - sat_2_tan_v)

        #if dv_tan < self.MIN_TAN_DV:
        #    self.MIN_TAN_DV = dv_tan
        # ------------ For plotting at end of training ------------ #


        s1_alt , _ , s1_a , s1_b = self.env_observe_state()
        gg_dist = np.sqrt(s1_a**2 + s1_b**2)

        if gg_dist < self.MIN_GG_DIST and 135 < s1_alt < 145:
            self.MIN_GG_DIST = gg_dist
        



    def env_observe_state(self):
        s1_alt = self.satellite_1.get_altitude(self.earth)
        dist = np.linalg.norm(self.satellite_1.position - self.satellite_2.position)

        # Orbital mechanics
        s1_r = np.linalg.norm(self.satellite_1.position - self.earth.position)
        s1_v = self.satellite_1.velocity
        s1_a = calc_semi_major_axis(v=s1_v , r=s1_r , c_obj=self.earth)
        s1_e = calc_eccentricity(v=s1_v , r=s1_r , c_obj=self.earth)
        s1_b = calc_semi_minor_axis(a=s1_a , e=s1_e)

        
        return (s1_alt , dist , s1_a , s1_b)


    
    def calculate_reward(self , state , action , next_state):
        s1_alt , dist , s1_a , s1_b = state
        next_s1_alt , next_dist , next_s1_a , next_s1_b = next_state
        reward = 0

        
        # Start : a = b = low alt
        # Part 1: low alt, get a -> target alt
        # Part 2: high alt, get b -> target alt
        # End   : a = b = high alt

        axis_thr = 10
        target_r = 140 + self.earth_radius

        """boost_reward = 0.001
        
        

        
        # Low alt
        if 55 < s1_alt < 65:
            dist_to_target = target_r - next_s1_a

            if abs(dist_to_target) > axis_thr:
                # We are not there (should boost)

                # We are too slow:
                if dist_to_target >= 0:
                    if action == 1: # +boost
                        reward += boost_reward
                    else:
                        reward -= 0

                # We are too fast:
                elif dist_to_target < 0:
                    if action == 2: # -boost
                        reward -= boost_reward
                    else:
                        reward -= 0

                
            else:
                # We are there (should wait)
                
                if action == 0: # wait
                    reward += 0
                else:
                    reward -= 0

        

        # High alt
        elif 135 < s1_alt < 140:

            dist_to_target = target_r - next_s1_b

            if abs(dist_to_target) > axis_thr:
                # We are not there (should boost)

                # We are too slow:
                if dist_to_target >= 0:
                    if action == 1: # +boost
                        reward += boost_reward
                    else:
                        reward -= 0

                # We are too fast:
                elif dist_to_target < 0:
                    if action == 2: # -boost
                        reward += boost_reward
                    else:
                        reward -= 0

                
            else:
                # We are there (should wait)
                
                if action == 0: # wait
                    reward += boost_reward
                else:
                    reward -= 0


        elif action == 1 or action == 2:
            reward -= 10"""
        
        if 70 < next_s1_alt < 130:
            if action == 1 or action == 2:
                reward -= 10
        else:
            if action == 1 or action == 2:
                if self.fuel < 99:
                    reward -= 1


        #if abs(target_r - next_s1_a) < axis_thr:
        #    print("-----obj 1")
        #if abs(target_r - next_s1_b) < axis_thr:
        #    print("-----obj 2")
        if abs(target_r - next_s1_a) < axis_thr and abs(target_r - next_s1_b) < axis_thr and action == 0 and 135 < next_s1_alt < 145:
            print("----------obj 3")
            reward += 50

        # Reach objective
        if next_dist < self.reached_dist and abs(target_r - next_s1_a) < axis_thr and abs(target_r - next_s1_b) < axis_thr:
            print("----------------GGs")
            reward += 2000
            self.goal_is_done = True

        


        # Crash
        if next_s1_alt <= 0 or self.fuel <= 0:
            print("crash reward" , ep_count)
            #reward -= 150

        # Go too far
        if next_s1_alt > 160:
            print("too far reward")
            #reward -= 150

        return reward


    

    def is_terminal(self , state):
        s1_alt , dist , s1_a , s1_b = state

        #print("terminal" , self.current_time_step)
        if s1_alt <= 0:
            print("terminal crash" , ep_count)
            return True
        elif self.fuel <= 0:
            print("terminal fuel" , ep_count)
            return True
        elif s1_alt > 160:
            print("terminal far" , ep_count)
            return True
        elif s1_a > 250 or s1_b > 250 or s1_a < 150 or s1_b < 150:
            return True

        return False




    def perform_action(self , a):
        global action_1_is_on , action_2_is_on
        
        # Observe current state
        current_state = self.env_observe_state()

        #print("ACTION" , a)

        
        
        

        # Perform action
        if a == 1 and self.goal_is_done == False:
            self.action_done += 1
            self.fuel -= 1
            self.satellite_1.change_tangent_velocity(self.earth , self.boost_strength)
            action_1_is_on = True
            self.satellite_1.position_where_thrust.append(self.satellite_1.position)
        elif a == 2:
            self.fuel -= 1
            self.satellite_1.change_tangent_velocity(self.earth , -self.boost_strength)
            action_2_is_on = True
            self.satellite_1.position_where_thrust2.append(self.satellite_1.position)
        elif a == 4:
            # 4: set velocity to stay in orbit 
            self.satellite_1.set_circular_orbit_velocity(self.earth , calculate_distance(self.satellite_1.position , self.earth.position))
        else:
            action_1_is_on = False
            action_2_is_on = False
            
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


        # Altitude subplot
        sat_1_alt = self.satellite_1.get_altitude(self.earth)
        sat_2_alt = self.satellite_2.get_altitude(self.earth)
        self.log_alt_sat_1.append(sat_1_alt)
        self.log_alt_sat_2.append(sat_2_alt)
        if action_1_is_on:
            self.log_alt_boost_pos.append(sat_1_alt)
        else:
            self.log_alt_boost_pos.append(np.nan)

        if action_2_is_on:
            self.log_alt_boost_neg.append(sat_1_alt)
        else:
            self.log_alt_boost_neg.append(np.nan)
        self.log_alt_time.append(self.current_time_step)


        # Trajectory subplot
        sat_1_pos = self.satellite_1.position
        sat_2_pos = self.satellite_2.position
        self.log_traj_sat_1.append(sat_1_pos)
        self.log_traj_sat_2.append(sat_2_pos)
        if action_1_is_on:
            self.log_traj_boost_pos.append(sat_1_pos)
        else:
            self.log_traj_boost_pos.append([np.nan , np.nan])

        if action_2_is_on:
            self.log_traj_boost_neg.append(sat_1_pos)
        else:
            self.log_traj_boost_neg.append([np.nan , np.nan])

        # Distance subplot
        self.log_distance.append(np.linalg.norm(self.satellite_1.position - self.satellite_2.position))
        
        # Radial velocity plot
        #self.log_radial_v.append(self.satellite_1.get_normal_velocity(self.earth))

        # Tangent velocity plot
        #self.log_tan_v_sat_1.append(self.satellite_1.get_tangent_velocity(self.earth))
        #self.log_tan_v_sat_2.append(self.satellite_2.get_tangent_velocity(self.earth))


        self.current_time_step += 1


    def plot_alts(self , ep):
        to_plot = 3

        if to_plot >= 1:
            plt.subplot(1, to_plot, 1)
            plt.plot(*zip(*self.log_traj_sat_1) , linestyle='-', color='b' , label='sat_1')
            plt.plot(*zip(*self.log_traj_sat_2) , linestyle='-', color='g' , label='sat_2')
            plt.plot(*zip(*self.log_traj_boost_pos) , marker='o', color='g' , label='+ boost')
            plt.plot(*zip(*self.log_traj_boost_neg) , marker='o', color='r' , label='- boost')

            circle = plt.Circle((500, 400), radius=100, edgecolor='blue', facecolor='blue')
            plt.gca().add_patch(circle)
            circle_o = plt.Circle((500, 400), radius=160, edgecolor='lightgray', facecolor='none')
            plt.gca().add_patch(circle_o)

            plt.xlabel('x')
            plt.ylabel('y')
            plt.axis('equal')
            plt.gca().invert_yaxis()
            plt.title('Trajectory')
            plt.legend()


        
        x = self.log_alt_time
        

        if to_plot >= 2:
            y1 = self.log_alt_sat_1
            y2 = self.log_alt_sat_2
            y3 = self.log_alt_boost_pos
            y4 = self.log_alt_boost_neg
            plt.subplot(1, to_plot, 2)
            plt.plot(x, y1 , linestyle='-', color='b' , label='sat_1')
            plt.plot(x, y2 , linestyle='-', color='g' , label='sat_2')
            plt.plot(x, y3, marker='o' , color='g' , label='+ boost')
            plt.plot(x, y4, marker='o' , color='r' , label='- boost')
            plt.xlabel('Time Step')
            plt.ylabel('Altitude')
            plt.title('Altitude')
            plt.legend()

        if to_plot >= 3:
            plt.subplot(1, to_plot, 3)
            plt.plot(x , self.log_distance)
            plt.xlabel('Time Step')
            plt.ylabel('Distance')
            plt.title('Distance')

        if to_plot >= 4:
            plt.subplot(1, to_plot, 4)
            plt.plot(x , self.log_radial_v)
            plt.xlabel('Time Step')
            plt.ylabel('Radial velocity')
            plt.title('Radial velocity')

        if to_plot >= 5:
            plt.subplot(1, to_plot, 5)
            plt.plot(x , self.log_tan_v_sat_1 , linestyle='-', color='b' , label='sat_1')
            plt.plot(x , self.log_tan_v_sat_2 , linestyle='-', color='g' , label='sat_2')
            plt.xlabel('Time Step')
            plt.ylabel('Tangent velocity')
            plt.title('Tangent velocity')


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
        return self.MIN_RAD_V_IN_REACH
    
    def get_min_dv_tan(self):
        return self.MIN_TAN_DV
    
    def get_action_done(self):
        return self.action_done
    
    def pass_count(self , message):
        global ep_count
        ep_count = message

    def get_the_plot(self):
        return self.MIN_GG_DIST



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
        for x in self.satellite_1.position_where_thrust:
            pygame.draw.circle(screen, (0, 255, 0), x, 2)
        for x in self.satellite_1.position_where_thrust2:
            pygame.draw.circle(screen, (255, 0, 0), x, 2)


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
        label(f"Fuel: {self.fuel}" , (self.width - 300 , 180) , screen)
        
        # Some orbital mechanics
        s1_r_orbit = np.linalg.norm(satellite_1.position - earth.position)
        s1_a = calc_semi_major_axis(v=satellite_1.velocity,r=s1_r_orbit,c_obj=self.earth)
        s1_e = calc_eccentricity(v=satellite_1.velocity,r=s1_r_orbit,c_obj=self.earth)
        s1_b = calc_semi_minor_axis(a=s1_a , e=s1_e)
        label(f"a: {round( s1_a ,2)}" , (self.width - 300, 140) , screen)
        label(f"b: {round( s1_b ,2)}" , (self.width - 170, 140) , screen)

        pygame.display.flip()
        self.clock.tick(60)