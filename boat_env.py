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
        Type: Box(4)
        Num	Observation                 Min         Max
        0	Cart Position             -4.8            4.8
        1	Cart Velocity             -Inf            Inf
        2	Pole Angle                 -24 deg        24 deg
        3	Pole Velocity At Tip      -Inf            Inf
        
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
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5 # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = 'euler'
        
        # Boat constants
        self.boatmass = 5000 # 5 tons
        self.boatlength = 11 # 11 meters
        self.boatbeam = 3 # 3 meters
        self.boatdraft = 1.2 # 1.2 meters
        self.max_rudder = 30 # 30 degres
        self.pos_rudder = -5 # 5 meters aft from CG
        self.boatspeed = 3 # 3m/s = 6kt, constant speed
        self.surfrud = 2 # 2 m2
        self.CzRud = 0.7
        self.boatIz = 46250 # cylinder m/4*(R2+h2/3) 
        self.rud_rate = 5 *2 * math.pi / 360 # rudders rate in rad/s
        
        
        # Water constants
        self.pho = 1.025 * 10**3# water density
        self.mi = 10**-3  # water viscosity

        # Angle at which to fail the episode
        self.theta_threshold_radians = 30 * 2 * math.pi / 360 # 12 * 2 * math.pi / 360
        self.x_threshold = 15 # 10 meters wide window - was 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        #high = np.array([
         #   self.x_threshold * 2,
          #  np.finfo(np.float32).max,
           # self.theta_threshold_radians * 2,
            #np.finfo(np.float32).max])

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
        state = self.state
        #x, x_dot, theta, theta_dot = state
        rud_rad, theta, theta_dot = state
        
        # modify the rudder position from action (if < max)
        if action == 0:
            rud_rad = min(rud_rad + self.rud_rate*self.tau, self.max_rudder * math.pi/180)
        else:
            rud_rad = max(rud_rad - self.rud_rate*self.tau, - self.max_rudder * math.pi/180)
        
        #force = self.force_mag if action==1 else -self.force_mag
        
        # hydro torque (gamma = 0, no drift )
        
        #C6 = -self.lp / self.L * self.Cy * np.sin(gamma) * np.abs(np.sin(gamma))
        #C6 = C6 - np.pi * self.Draft / self.L * np.sin(gamma) * np.cos(gamma)
        #C6 = C6 - (0.5 + 0.5 * np.abs(np.cos(gamma))) ** 2 * np.pi * self.Draft / self.L * (0.5 - 2.4 * self.Draft / self.L) * np.sin(gamma) * np.abs(np.cos(gamma))
        #C6 = - np.pi * self.boatdraft / self.boatlength * (0.5 - 2.4 * self.boatdraft / self.boatlength) * np.sin(gamma) * np.abs(np.cos(gamma))
        #F1z = 0.5 * self.pho * self.boatspeed ** 2 * self.boatlength**2 * self.boatdraft * C6
        
        # rudder torque
        Crud = 0.5 * self.pho * self.boatspeed ** 2 * self.surfrud * self.CzRud * math.sin(rud_rad) * self.pos_rudder
        
        # d(theta_dot) = Crud / Iz
        theta_dotdot = - Crud / self.boatIz
        
        #force = self.force_mag if action==1 else -self.force_mag
        #costheta = math.cos(theta)
        #sintheta = math.sin(theta)
        #temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        #thetaacc = (self.gravity * sintheta - costheta* temp) / (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        #xacc  = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        
        if self.kinematics_integrator == 'euler':
           # x  = x + self.tau * x_dot
           # x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * theta_dotdot
        else: # semi-implicit euler
           # x_dot = x_dot + self.tau * xacc
           # x  = x + self.tau * x_dot
           # theta_dot = theta_dot + self.tau * thetaacc
           # theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * theta_dotdot
            theta = theta + self.tau * theta_dot
            
        self.state = (rud_rad,theta,theta_dot)
        
        done = theta < -self.theta_threshold_radians or theta > self.theta_threshold_radians
        done = bool(done)

        if not done:
            reward = - np.abs(theta) **2 # - np.abs(rud_rad) # 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 0.0 # 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.1, high=0.1, size=(3,))
        self.steps_beyond_done = None
        return np.array(self.state)

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
