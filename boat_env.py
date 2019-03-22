"""
Boat env derived from the gym cartpole
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

class  BoatEnv(gym.Env):
    """
    Description:
Boat at constant speed, only lateral forces and torques at the moment. Trying to keep the heading theta

    Source:
        This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson

    Observation: 
        Type: Box(3)
        Num	Observation                 Min         Max
        0	Rudder position           -30 deg        30 deg
        1	Boat heading theta        -Pi            Pi
        2	Theta dot                 -Inf           Inf
        
    Actions:
        Type: Discrete(2)
        Num	Action
        0	Action the rudder left
        1	Action the rudder right
        
        Note: The amount the velocity that is reduced or increased is not fixed; it depends on the angle the pole is pointing. This is because the center of gravity of the pole increases the amount of energy needed to move the cart underneath it

    Reward:
        Reward is 1 for every step taken, including the termination step

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

    def __init__(self):

        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = 'euler'
        self.simu = 'complex'
        
        # Boat constants
        self.boatmass = 5000 # 5 tons
        self.boatlength = 11 # 11 meters
        self.boatbeam = 3 # 3 meters
        self.boatdraft = 0.8 # 1.2 meters
        self.max_rudder = 30 # 30 degres
        self.pos_rudder = -5 # 5 meters aft from CG
        self.boatspeed = 3 # 3m/s = 6kt, constant speed
        self.surfrud = 2 # 2 m2
        self.CzRud = 0.7
        self.Izz = 46250 # cylinder m/4*(R2+h2/3) 
        self.rud_rate = 3 *2 * math.pi / 360 # rudder rate in rad/s
        
        self.x_midship = 0 # x-coord at mid ship
        self.a4 = 1
        self.a5 = 1
        self.b4 = 1
        self.c3 = 1
        self.Jzz = 0 # added moment of inertia zz
        
        self.lambd = 1.6 # rudder aspect ratio
        self.Ar = 2 # rudder area
        self.aH = 0 # ratio of hydrodynamic force, induced on hull by rudder action, to rudder force
        self.xR = -5 # x-coord of point on which lateral rudder force acts

        
        # Water constants
        self.pho = 1.025 * 10**3# water density
        self.mi = 10**-3  # water viscosity

        # Angle at which to fail the episode
        self.theta_threshold_radians = 60 * 2 * math.pi / 360 # 12 * 2 * math.pi / 360
        self.x_threshold = 15 # 10 meters wide window - was 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.max_rudder * math.pi/180,
            math.pi,
            np.finfo(np.float32).max])
              
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        
        # assign states
        state = self.state
        rud_rad, theta, theta_dot = state
        
        # modify the rudder position from action (if < max)
        if action == 1:
            rud_rad = min(rud_rad + self.rud_rate*self.tau, self.max_rudder * math.pi/180)
        else:
            rud_rad = max(rud_rad - self.rud_rate*self.tau, - self.max_rudder * math.pi/180)
        
        ###############################################################
        # Very simple simulation model, in air only with inertia torque
        if self.simu == 'simple':
       
            # rudder torque
            Crud = 0.5 * self.pho * self.boatspeed ** 2 * self.surfrud * self.CzRud * math.sin(rud_rad) * self.pos_rudder

            # d(theta_dot) = Crud / Iz (in air, only inertia effect)
            theta_dotdot = - Crud / self.Izz

            
        #####################################################################
        # yaw simu model from S.Inoue, no propeller, no drift, constant speed
        elif self.simu == 'complex':
            
            #############
            # Hull effect (v=0, v'=0, u'=0)
                     
            rprim = theta_dot*self.boatlength / self.boatspeed # dimensionless turning rate
            
            k = 2*self.boatdraft/self.boatlength
            
            tauprim = 1/self.boatdraft # tau (=1?)/draft : trim quantity
            
            Nrprim = (self.a4*k + self.a5*k**2)*(1 + self.b4*tauprim)
            
            Nrrprim = self.c3 * Nrprim
            
            Nh0 = - 0.5 * self.pho * self.boatlength * self.boatdraft * self.boatspeed**2 *( Nrprim * rprim + Nrrprim * rprim*np.abs(rprim) )
            
            Nh1 = 0 # no roll
            
            Yh1 = 0
            
            # Nh = -Jzz*theta_dotdot + Nh0 + Nh1 + (Yh0 + Yh1)*self.x_midship
            
            ###############
            # Rudder effect
                     
            Vr = self.boatspeed # rudder Speed ~ boat speed (prop effect?)
            
            alphaR = rud_rad # rudder aoa ~ rud angle
            
            Fn = 0.5 * self.pho * (6.13*self.lambd / (self.lambd + 2.25)) * self.Ar * Vr**2 * np.sin(alphaR)
            
            Nr = -(1 + self.aH) * self.xR * Fn * np.cos(rud_rad)
            
            ###############
            # Roll equation
                  
            # theta_dotdot = (Nh + Nr) / self.Izz
            
            # if Jzz = 0 and X-midship=0
            theta_dotdot = (Nh0 + Nr) / self.Izz
            
        ############
        # integrator
        
        if self.kinematics_integrator == 'euler':
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * theta_dotdot
        else: # semi-implicit euler
            theta_dot = theta_dot + self.tau * theta_dotdot
            theta = theta + self.tau * theta_dot
        
        self.state = (rud_rad,theta,theta_dot)
        
        done = theta < -self.theta_threshold_radians or theta > self.theta_threshold_radians
        done = bool(done)
        
        ########
        # reward
        if not done:
            if np.abs(theta) < 0.02:
                reward = 1
            else: 
                reward = 0
            #reward = math.cos(theta) **2 # - np.abs(rud_rad) # 1.0
        elif self.steps_beyond_done is None:
            # Boat out of track!
            self.steps_beyond_done = 0
            reward = -1.0 # 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}
    
    #######################
    # state initialisation
    def reset(self): 
        self.state = self.np_random.uniform(low=-0.1, high=0.1, size=(3,))
        self.steps_beyond_done = None
        return np.array(self.state)
   
    ###########
    # rendering
    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold*2
        scale = screen_width/world_width
        carty = 100 # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0
        
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
        self.rudderot.set_rotation(-x[0]*10) # rudder angle * 10 for display
 
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
