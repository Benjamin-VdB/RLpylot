"""
Boat env derived from the gym cartpole
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

import sys
import json
import logging
import requests
import threading
import time
import websocket # 0.48.0
import pkg_resources

from signalk_client.client import Client, Data
import datetime as dt

import serial
from smcg2serial import SmcG2Serial

class  BoatEnv(gym.Env):
    """
    Description:
Boat at constant speed, only lateral forces and torques at the moment. Trying to keep the heading theta to 0

    Source:
        This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson

    Observation: 
        Type: Box(4)
        Num	Observation                 Min         Max
        0	Rudder position           -30 deg        30 deg
        1	Boat heading theta        -Pi            Pi
        2	Theta dot                 -Inf           Inf
        (3	Boat speed                -Inf           Inf)
                
    Actions:
        Type: Discrete(5) or 10
        Num	Action
        0	Action the rudder left fast
        1	Action the rudder left slow
        2	No action on the rudder
        3	Action the rudder right slow
        4	Action the rudder right fast
        

    Reward:
        Reward is 1 when abs(theta) < 0.1 - cost for action taken

    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]

    Episode Termination:
        Pole Angle is more than 12 degrees
        Cart Position is more than 2.4 (center of the cart reaches the edge of the display)
        Episode length is greater than 200
        Solved Requirements
        Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.
    """
    
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }
    
    # heading target
    target = 0.

    # Environment initialisation
    def __init__(self, type='discrete', mode='simulation'):
        
        self.type = type
        self.mode = mode
        
        self.tau = 0.1  # seconds between state updates
        self.kinematics_integrator = 'semi'
        self.type_simu ='complex'
        
        # Boat constants
        self.boatmass = 5000. # 5 tons
        self.boatlength = 11. # 11 meters
        self.boatbeam = 3. # 3 meters
        self.boatdraft = 0.8 # 1.2 meters
        self.max_rudder = 30. # 30 degres
        self.pos_rudder = -5. # 5 meters aft from CG
        self.boatspeed = 3. # 3m/s = 6kt, constant speed
        self.surfrud = 2. # 2 m2
        self.CzRud = 0.7
        self.Izz = 46250. # cylinder m/4*(R2+h2/3) 
        self.rud_rate_low = 3 *2 * math.pi / 360 # rudder rate in rad/s
        self.rud_rate_high = 6 *2 * math.pi / 360 # rudder rate in rad/s
        
        self.max_rud_rate = 3. *2 * math.pi / 360 # max rudder rate in rad/s (reference 30deg !!!)
        self.max_motor_command = 3200 # max command sent to the motor
        self.rud_angle_ratio = 30/18 # measured angle vs. real angle
        
        self.x_midship = 0. # x-coord at mid ship
        self.a4 = 1.
        self.a5 = 1.
        self.b4 = 10. # 10 to increase the hull effect
        self.c3 = 10.
        self.Jzz = 0. # added moment of inertia zz
        
        self.lambd = 1.6 # rudder aspect ratio
        self.Ar = 2. # rudder area
        self.aH = 0. # ratio of hydrodynamic force, induced on hull by rudder action, to rudder force
        self.xR = -5. # x-coord of point on which lateral rudder force acts

        
        # Water constants
        self.pho = 1.025 * 10**3# water density
        self.mi = 10.**-3  # water viscosity

        # Angle at which to fail the episode
        self.theta_threshold_radians = 60. * math.pi / 180 # 12 * 2 * math.pi / 360
        self.x_threshold = 15. # 10 meters wide window - was 2.4

        # Observation space (rud_rad, theta, theta_dot, speed)
        high = np.array([
            self.max_rudder * math.pi/180,
            math.pi,
            1 ]) 
            #, np.finfo(np.float32).max])
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)   
        
        # Action space
        if self.type == 'discrete':
            self.action_space = spaces.Discrete(5)
            
        elif self.type == 'continuous':
            self.action_space = spaces.Box(low=-self.max_rud_rate, high=self.max_rud_rate, shape=(1,), dtype=np.float32)

            
        # Real world init
        if self.mode == 'real':
            # list for memory
            thetatime_list = []
            theta_list = []
            thetadot_list = []
            
            # Signal K client init
            sk_client = Client(server='localhost:3000') # RPi at 192.168.0.108
            vessel = sk_client.data.get_self()
            
            # Motor init
            port_name = "/dev/ttyACM0"  # 'COM3' from windows 
            baud_rate = 9600
            device_number = None
            port = serial.Serial(port_name, baud_rate, timeout=0.1, write_timeout=0.1)
            smc = SmcG2Serial(port, device_number)
            smc.exit_safe_start()
                    
                        
        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
 
    
    ################################
    # Simulation function from action
    
    def simulation(self, action):
        
        type_simu = self.type_simu
               
        # assign states
        rud_rad, theta, theta_dot= self.state # ,speed
    
        ##############################################################
        # modify the rudder position from action (if < max)
        # discrete action
        if self.type == 'discrete':
            new_rud_rad = np.clip(rud_rad + (2*action/(self.action_space.n -1) -1)*self.max_rud_rate*self.tau, -self.max_rudder * math.pi/180, self.max_rudder * math.pi/180)
        
        # continuous action on the rudder rate
        elif self.type == 'continuous':
            #rud_rad = max(min(rud_rad + self.rud_rate_high*self.tau * action, self.max_rudder * math.pi/180), - self.max_rudder * math.pi/180)
            action = np.clip(action, -self.max_rud_rate, self.max_rud_rate)[0]
            new_rud_rad = np.clip(rud_rad + action*self.tau, -self.max_rudder*np.pi/180, self.max_rudder*np.pi/180)
    
    
        ###############################################################
        # Very simple simulation model, in air only with inertia torque
        if type_simu == 'simple':
       
            # rudder torque
            Crud = 0.5 * self.pho * self.boatspeed ** 2 * self.surfrud * self.CzRud * math.sin(rud_rad) * self.pos_rudder
            # d(theta_dot) = Crud / Iz (in air, only inertia effect)
            theta_dotdot = - Crud / self.Izz

            
        #####################################################################
        # yaw simu model from S.Inoue, no propeller, no drift, constant speed
        if type_simu == 'complex':
            
            speed = self.boatspeed
            
            #############
            # Hull effect (v=0, v'=0, u'=0)      
            rprim = theta_dot*self.boatlength / speed # dimensionless turning rate
            k = 2*self.boatdraft/self.boatlength
            tauprim = 1/self.boatdraft # tau (=1?)/draft : trim quantity
            Nrprim = (self.a4*k + self.a5*k**2)*(1 + self.b4*tauprim)
            Nrrprim = self.c3 * Nrprim
            Nh0 = - 0.5 * self.pho * self.boatlength * self.boatdraft * speed**2 *( Nrprim * rprim + Nrrprim * rprim*np.abs(rprim) )
            Nh1 = 0 # no roll
            Yh1 = 0
            # Nh = -Jzz*theta_dotdot + Nh0 + Nh1 + (Yh0 + Yh1)*self.x_midship
            
            ###############
            # Rudder effect   
            Vr = speed # rudder Speed ~ boat speed (prop effect?)
            alphaR = new_rud_rad # rudder aoa ~ rud angle
            Fn = 0.5 * self.pho * (6.13*self.lambd / (self.lambd + 2.25)) * self.Ar * Vr**2 * np.sin(alphaR)
            Nr = -(1 + self.aH) * self.xR * Fn * np.cos(new_rud_rad)
            
            ####################
            # wind effect
            #Nw = 
            
            ###############
            # Yaw equation
            # theta_dotdot = (Nh + Nr) / self.Izz
            # if Jzz = 0 and X-midship=0
            new_theta_dotdot = (Nh0 + Nr) / self.Izz
            
            #print("Nh0: ", Nh0)
            #print("rudder : ", rud_rad)
            #print("Nr: ", Nr) # 10000 for 0.1 rad
            
        ############
        # integrator
        
        if self.kinematics_integrator == 'euler':
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * theta_dotdot
        else: # semi-implicit euler
            new_theta_dot = theta_dot + self.tau * new_theta_dotdot
            #new_theta_dot = np.clip(new_theta_dot, -1., 1.)
            new_theta = theta + self.tau * new_theta_dot
        
        
        #############    
        # simu output
        return np.array([new_rud_rad,new_theta,new_theta_dot])
    
    
    #######################
    # real world function from action
    
    def real_step(self, action):

        # take action on the motor if abs(rud_rad) < 25 / should we limit the action to >2 ?
        if np.abs(vessel.get_prop('steering.rudderAngle')['value'] * self.rud_angle_ratio) < 25*math.pi/ 180:
            smc.set_target_speed((2*action/(self.action_space.n -1) -1)*self.max_motor_command)
        else:
            smc.set_target_speed(0)

        # get the states from signal K
        theta = angle_normalize(vessel.get_prop('navigation.headingMagnetic')['value'] - target)
        thetatime = dt.datetime.strptime(vessel.get_prop('navigation.headingMagnetic')['timestamp'], '%Y-%m-%dT%H:%M:%S.%fZ').timestamp()
        rud_rad = vessel.get_prop('steering.rudderAngle')['value'] * self.rud_angle_ratio

        theta_list.append(theta)
        thetatime_list.append(thetatime)

        # calculate thetadot based on the 4th previous observation (~2s for a 2Hz signal)
        if len(theta_list) >5:
            theta_dot = (theta_list[-1] - theta_list[-4]) / (thetatime_list[-1] - thetatime_list[-4])
        else:
            thetadot = 0
        thetadot_list.append(theta_dot)


        # wait for the action results and observe the new states

        time.sleep(0.5)

        new_theta = angle_normalize(vessel.get_prop('navigation.headingMagnetic')['value'] - target)
        new_thetatime = dt.datetime.strptime(vessel.get_prop('navigation.headingMagnetic')['timestamp'], '%Y-%m-%dT%H:%M:%S.%fZ').timestamp()
        new_rud_rad = vessel.get_prop('steering.rudderAngle')['value'] * self.rud_angle_ratio

        theta_list.append(new_theta)
        thetatime_list.append(new_thetatime)

        # calculate thetadot based on the 4th previous observation (~2s for a 2Hz signal)
        if len(theta_list) >5:
            new_theta_dot = (theta_list[-1] - theta_list[-4]) / (thetatime_list[-1] - thetatime_list[-4])
        else:
            new_thetadot = 0
        thetadot_list.append(new_thetadot)

        # Limit lists to 20 items to avoid keeping too much data
        thetatime_list = thetatime_list[-20:]
        theta_list = theta_list[-20:]
        thetadot_list = thetadot_list[-20:]
    
        #############    
        # world output
        return np.array([new_rud_rad,new_theta,new_theta_dot])
    
    
    
    
    #######################
    # Reward function
    def reward(self, obs, action):
        rud_rad, theta, thetadot = obs[0], obs[1], obs[2]
        
        if not self.done:
            if self.type == 'discrete':
                #reward = 10*math.cos(theta) **2 - .1*theta_dot**2 - .0001*((action-5)/5) **2
                #reward = 1/np.abs(theta) - .1*theta_dot**2 - .01*((action-5)/5) **2
                if np.abs(theta) < 0.01:
                    return 1. - 0.1*np.abs((2*action/(self.action_space.n -1) -1)) 
                else: 
                    return - 0.1*np.abs((2*action/(self.action_space.n -1) -1))
                # return - (theta)**2 - .1*thetadot**2 - .001*(((2*action/(self.action_space.n -1) -1)**2)) # doesn't work

            elif self.type == 'continuous':
                if np.abs(theta) < 0.02:
                    return 1 - 0.1*(action/self.max_rud_rate) **2
                else:
                    return 0. #math.cos(theta) **2
                #reward = math.cos(theta) **2 - .1*theta_dot**2 - .001*((action-5)/5) **2
                #if np.abs(theta) < 0.02:
                    #reward = 1
                    #reward = math.cos(theta) **2 - .1*theta_dot**2 - .001*action**2
                #else:
                    #reward = 0
                    #reward = math.cos(theta) **2 - .1*theta_dot**2 - .001*action**2
                    
        elif self.steps_beyond_done is None:
            # Boat out of track!
            self.steps_beyond_done = 0
            return -1. # 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            return 0.0

    def angle_normalize(x):
        return (((x+np.pi) % (2*np.pi)) - np.pi)
    
    ###################
    # RL step function
    def step(self, action):

        # Get the next state from action
        if self.mode == 'simulation':
            self.state = self.simulation(action)
        
        if self.mode == 'real':
            self.state = self.real_step(action)
                       
        self.action = action
        
        # If the episode done?
        self.done = self.state[1] < -self.theta_threshold_radians or self.state[1] > self.theta_threshold_radians
        self.done = bool(self.done)
        
        # Get the reward for the episode
        rwd = self.reward(self.state, action)

        return np.array(self.state), rwd, self.done, {}
    

    

    #######################
    # state initialisation
    def reset(self): 
        
        thetatime_list = []
        theta_list = []
        thetadot_list = []

        if self.mode == 'simulation':
            #self.state = self.np_random.uniform(low=-0.1, high=0.1, size=(3,))
            self.state = np.concatenate( (self.np_random.uniform(low=-0.2, high=0.2, size=(1,)),   # rud init
                                          self.np_random.uniform(low=-0.6, high=0.6, size=(1,)),   # theta init
                                          self.np_random.uniform(low=-0.05, high=0.05, size=(1,)))) #, # thetadot init
                                          #self.np_random.uniform(low=1.5, high=4, size=(1,))) ) # speed init
            
            
        if self.mode == 'real':
            rud_rad = vessel.get_prop('steering.rudderAngle')['value']
            theta = angle_normalize(vessel.get_prop('navigation.headingMagnetic')['value'] - target)
            thetadot = 0
            
            self.state = np.array([rud_rad,theta,theta_dot]) #,speed])          
            
        self.steps_beyond_done = None
        return np.array(self.state)
   


    ###########
    # rendering
    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold*2
        scale = screen_width/world_width
        
        boatlen = scale * self.boatlength
        boatwidth = scale * self.boatbeam
        rudderlen = 2 * scale
        rudderwidth = 0.5 * scale
        
        
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            # boat rendering
            l,r,t,b = -boatwidth/2, boatwidth/2, boatlen/2, -boatlen/2
            axleoffset = boatlen/4.0
            boat = rendering.FilledPolygon([(l,b), (l,t*1/4), (0, t), (r,t*1/4), (r,b)])
            self.boatrot = rendering.Transform()
            boat.add_attr(self.boatrot)
            self.viewer.add_geom(boat)

            
            # rudder rendering
            l,r,b = -rudderwidth/2, rudderwidth/2, -rudderlen
            rudder = rendering.FilledPolygon([(l,0), (r,0), (0,b)])
            rudder.set_color(1,.0, 0)
            self.rudderot = rendering.Transform(translation=(0, -boatlen/2))
            rudder.add_attr(self.rudderot)
            rudder.add_attr(self.boatrot)
            self.viewer.add_geom(rudder)
            
            #self.axle = rendering.make_circle(polewidth/2)
            #self.axle.add_attr(self.poletrans)
            #self.axle.add_attr(self.carttrans)
            #self.axle.set_color(.5,.5,.8)
            #self.viewer.add_geom(self.axle)
            #self.track = rendering.Line((0,carty), (screen_width,carty))
            #self.track.set_color(0,0,0)
            #self.viewer.add_geom(self.track)
            
            # params rendering
            #theta = rendering.text.Label(x[1],
             #             font_name='Times New Roman',
              #            font_size=36,
               #           x=screen_width/4, y=screen_height/2,
                #          anchor_x='center', anchor_y='center')

            self._rudder_geom = rudder

        if self.state is None: return None

        # Edit the rudder polygon vertex
        rudder = self._rudder_geom
        l,r,b = -rudderwidth/2, rudderwidth/2, -rudderlen
        rudder.v = [(l,0), (r,0), (0,b)]

        x = self.state
        
        #cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
        self.boatrot.set_translation(screen_width/2, screen_height/2)
        self.boatrot.set_rotation(x[1])
        self.rudderot.set_rotation(-x[0]*3) # rudder angle * 3 for display
 
        return self.viewer.render(return_rgb_array = mode=='rgb_array')



    def close(self):
        if self.mode == 'real':
            # close signal K client
            sk_client.close()

            # stop the motor
            smc.set_target_speed(0)
        
        if self.viewer:
            self.viewer.close()
            self.viewer = None
