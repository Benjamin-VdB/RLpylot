#!/usr/bin/python
#-*- coding: utf-8 -*-
import numpy as np
from scipy.integrate import RK45


class Simulator:
    def __init__(self):
        self.last_global_state = None
        self.last_local_state = None
        self.current_action = None
        self.steps = 0
        self.time_span = 10           # 10 seconds for each iteration
        self.number_iterations = 100  # 100 iterations for each step
        self.integrator = None
        self.rk_mode = 'scipy_rk'

        ##Vessel Constants

        self.M = 115000      *10**3
        self.Iz = 414000000 * 10 ** 3

        self.M11 = 14840.4   * 10**3
        self.M22 = 174050    * 10**3
        self.M26 = 38369.6   * 10**3
        self.M66 = 364540000 * 10**3
        self.M62 = 36103 * 10**3

        self.D11 = 0.35370 * 10**3
        self.D22 = 1.74129 * 10**3
        self.D26 = 1.95949 * 10**3
        self.D62 = 1.85586 * 10**3
        self.D66 = 3.23266 * 10**3

        self.L = 244.74 #length
        self.Draft = 15.3
        self.x_g = 2.2230# center mass
        self.x_prop = -112 #propulsor position
        self.force_prop_max = 1.6 * 10**6 # max porpulsor force
        self.x_rudder = -115 # rudder position
        self.rudder_area = 68

        self.Cy = 0.06           # coeff de arrasto lateral
        self.lp = 7.65 # cross-flow center
        self.Cb = 0.85           # block coefficient
        self.B = 42             # Beam
        self.S = 27342        # wet surface

        ## Water constants
        self.pho = 1.025 * 10**3# water density
        self.mi = 10**-3  # water viscosity

        ## Rudder Constants
        self.A_rud = 68 # propulsor thrus
        self.delta_x = self.x_prop - self.x_rudder  # distance between rudder and propulsor
        self.r_aspect = 2 # aspect ration

        ## Propulsor constants:
        self.D_prop = 7.2 # Diameter
        self.n_prop = 1.6 # rotation

        # some modes of simulator
        self.system_dynamics = 'complex'
        self.prop_dynamics = 'complex'


    def reset_start_pos(self, global_vector):
        x0, y0, theta0, vx0, vy0, theta_dot0 = global_vector[0], global_vector[1], global_vector[2], global_vector[3], global_vector[4], global_vector[5]
        self.last_global_state = np.array([x0, y0, theta0, vx0, vy0, theta_dot0])
        self.last_local_state = self._global_to_local(self.last_global_state)
        if self.rk_mode == 'scipy_rk':
            self.current_action = np.zeros(2)
            self.integrator = self.scipy_runge_kutta(self.simulate_scipy, self.get_state(), t_bound=self.time_span)

    def step(self, angle_level, rot_level):
        self.current_action = np.array([angle_level, rot_level])
        if self.rk_mode == 'ours_rk':
            for i in range(self.number_iterations):
                self.last_global_state = self.runge_kutta(self.get_state(), self.simulate_in_global, 6, self.time_span/self.number_iterations)
            return self.last_global_state

        if self.rk_mode == 'scipy_rk':
            while not (self.integrator.status == 'finished'):
                self.integrator.step()

            self.last_global_state = self.integrator.y
            self.last_local_state = self._global_to_local(self.last_global_state)
            self.integrator = self.scipy_runge_kutta(self.simulate_scipy, self.get_state(), t0=self.integrator.t, t_bound=self.integrator.t+self.time_span)
            return self.last_global_state

    def simulate_scipy(self, t, global_states):
        local_states = self._global_to_local(global_states)
        return self._local_ds_global_ds(global_states[2], self.simulate(local_states))

    def simulate_in_global(self, global_states):
        local_states = self._global_to_local(global_states)
        return self._local_ds_global_ds(global_states[2], self.simulate(local_states))

    def simulate(self, local_states):
        """
        :param local_states: Space state
        :return df_local_states
        """
        x1 = local_states[0] #u
        x2 = local_states[1] #v
        x3 = local_states[2] #theta (not used)
        x4 = local_states[3] #du
        x5 = local_states[4] #dv
        x6 = local_states[5] #dtheta
        beta = self.current_action[0]*np.pi/6   #leme (-30 à 30)
        alpha = self.current_action[1]    #propulsor

        vc = np.sqrt(x4 ** 2 + x5 ** 2)
        gamma = np.pi+np.arctan2(x5, x4)

        # Composing resistivity forces
        Re = self.pho * vc * self.L / self.mi
        if Re == 0:
            C0=0
        else:
            C0 = 0.0094 * self.S / (self.Draft * self.L) / (np.log10(Re) - 2) ** 2
        C1 = C0 * np.cos(gamma) + (-np.cos(3 * gamma) + np.cos(gamma)) * np.pi * self.Draft / (8 * self.L)
        F1u = 0.5 * self.pho * vc ** 2 * self.L * self.Draft * C1

        C2 = (self.Cy - 0.5 * np.pi * self.Draft / self.L) * np.sin(gamma) * np.abs(np.sin(gamma)) + 0.5 * np.pi * self.Draft / self.L * (
                np.sin(gamma) ** 3) + np.pi * self.Draft / self.L * (1 + 0.4 * self.Cb * self.B / self.Draft) * np.sin(gamma) * np.abs(np.cos(gamma))
        F1v = 0.5 * self.pho * vc ** 2 * self.L * self.Draft * C2

        C6 = -self.lp / self.L * self.Cy * np.sin(gamma) * np.abs(np.sin(gamma))
        C6 = C6 - np.pi * self.Draft / self.L * np.sin(gamma) * np.cos(gamma)
        C6 = C6 - (0.5 + 0.5 * np.abs(np.cos(gamma))) ** 2 * np.pi * self.Draft / self.L * (0.5 - 2.4 * self.Draft / self.L) * np.sin(gamma) * np.abs(np.cos(gamma))
        F1z = 0.5 * self.pho * vc ** 2 * self.L**2 * self.Draft * C6

        # Propulsion model
        if self.prop_dynamics == 'simple':
            Fpx = np.cos(beta) * self.force_prop_max * alpha * np.abs(2/(1+x1))
            Fpy = -np.sin(beta) * self.force_prop_max * alpha * np.abs(2/(1+x1))
            Fpz = Fpy * self.x_rudder
        else:
            #Propulsion model complex -- > the best one:
            J = x4*0.6/(1.6*7.2)
            kt = 0.5 - 0.5*J
            n_prop_ctrl = self.n_prop*alpha
            Fpx = kt*self.pho*n_prop_ctrl**2*self.D_prop**4

            kr = 0.5 + 0.5 / (1 + 0.15 * self.delta_x/self.D_prop)
            ur = np.sqrt(x4 ** 2 + kr * 4 * kt * n_prop_ctrl ** 2 * self.D_prop ** 2 / np.pi)
            vr = -0.8 * x5
            Ur = np.sqrt(ur ** 2 + vr ** 2)
            fa = 6.13 * self.r_aspect / (self.r_aspect + 2.25)
            ar = beta
            FN = 0.5*self.pho*self.A_rud*fa*Ur**2*np.sin(ar)
            Fpy = -FN * np.cos(beta)
            Fpz = -FN * np.cos(beta) * self.x_rudder


        # without resistence
        #F1u, F1v, F1z = 0, 0, 0
        # Derivative function

        fx1 = x4
        fx2 = x5
        fx3 = x6

        # simple model
        if self.system_dynamics == 'complex':
            Mrb = np.array([[self.M, 0, 0], [0, self.M, self.M*self.x_g], [0, self.M*self.x_g, self.Iz]])
            Crb = np.array([[0, 0, -self.M*(self.x_g*x6+x5)], [0, 0, self.M*x4], [self.M*(self.x_g*x6+x5), -self.M*x4, 0]])
            Ma  = np.array([[self.M11, 0, 0], [0, self.M22, self.M26], [0, self.M62, self.M66]])

            ca13 = -(self.M22*x5 + self.M26*x6)
            ca23 = self.M11*x4
            Ca  = np.array([[0, 0, ca13], [0, 0, ca23], [-ca13, -ca23, 0]])
            Dl = np.array([[self.D11, 0, 0], [0, self.D22, self.D26], [0, self.D62, self.D66]])
            vv = np.array([x4, x5, x6])


            MM = Mrb+Ma
            CC = Crb+Dl

            Fext = np.array([[F1u + Fpx], [F1v + Fpy], [0.21*F1z + 0.5*Fpz]])
            A = MM
            B = np.dot(CC, vv.transpose()) + Fext.transpose()

            ff = np.linalg.solve(A, B.transpose())

            fx4 = ff[0]
            fx5 = ff[1]
            fx6 = ff[2]
        elif self.system_dynamics == 'linearized':
            a11 = self.M + self.M11
            b1 = -(self.M + self.M22) * x5 * x6 - (self.M * self.x_g + 0.5 * (self.M26 + self.M62)) * x6 ** 2

            fx4 = (b1+F1u + Fpx)/ a11

            A = np.array([[self.M + self.M26, self.M * self.x_g + self.M22], [self.M * self.x_g + self.M62, self.Iz + self.M66]])

            B1 = [[self.D26, self.M * x4 + self.D22], [self.M62, self.x_g * x4 + self.D66]]

            vv = np.array([x5, x6])
            Fext = np.array([[F1v + Fpy], [F1z + Fpz]])

            B = np.dot(B1, vv.transpose()) + Fext.transpose()
            ff = np.linalg.solve(A, B.transpose())

            fx5 = ff[0]
            fx6 = ff[1]
        else:
            # main model simple -- > the best one:
            fx4 = (F1u + Fpx)/(self.M + self.M11)
            fx5 = (F1v + Fpy)/(self.M + self.M22)
            fx6 = (F1z + Fpz)/(self.Iz + self.M66)

        fx = np.array([fx1, fx2, fx3, fx4, fx5, fx6])
        return fx

    def scipy_runge_kutta(self, fun, y0, t0=0, t_bound=10):
        return RK45(fun, t0, y0, t_bound,  rtol=self.time_span/self.number_iterations, atol=1e-4)

    def runge_kutta(self, x, fx, n, hs):
        k1 = []
        k2 = []
        k3 = []
        k4 = []
        xk = []
        ret = np.zeros([n])
        for i in range(n):
            k1.append(fx(x)[i]*hs)
        for i in range(n):
            xk.append(x[i] + k1[i]*0.5)
        for i in range(n):
            k2.append(fx(xk)[i]*hs)
        for i in range(n):
            xk[i] = x[i] + k2[i]*0.5
        for i in range(n):
            k3.append(fx(xk)[i]*hs)
        for i in range(n):
            xk[i] = x[i] + k3[i]
        for i in range(n):
            k4.append(fx(xk)[i]*hs)
        for i in range(n):
            ret[i] = x[i] + (k1[i] + 2*(k2[i] + k3[i]) + k4[i])/6
        return ret

    def get_state(self):
        return self.last_global_state

    def get_local_state(self):
        return self.last_local_state

    def _local_to_global(self, local_state):
        # local_state: [ux, uy, theta, uxdot, uydot, thetadot]
        theta = local_state[2]
        c, s = np.cos(theta), np.sin(theta)
        A = np.array([[c, -s], [s, c]])
        B_l_pos = np.array([local_state[0], local_state[1]])
        B_l_vel = np.array([local_state[3], local_state[4]])

        B_g_pos = np.dot(A, B_l_pos.transpose())
        B_g_vel = np.dot(A, B_l_vel.transpose())
        return np.array([B_g_pos[0], B_g_pos[1], local_state[2], B_g_vel[0], B_g_vel[1], local_state[5]])

    def _global_to_local(self, global_state):
        # global_states: [x, y, theta, vx, vy, thetadot]
        theta = global_state[2]
        c, s = np.cos(theta), np.sin(theta)
        A = np.array([[c, s], [-s, c]])
        B_g_pos = np.array([global_state[0], global_state[1]])
        B_g_vel = np.array([global_state[3], global_state[4]])

        #print("global",global_state,A,B_g_pos,B_g_vel)
        B_l_pos = np.dot(A, B_g_pos.transpose())
        B_l_vel = np.dot(A, B_g_vel.transpose())
        return np.array([B_l_pos[0], B_l_pos[1], global_state[2], B_l_vel[0], B_l_vel[1], global_state[5]])

    def _local_ds_global_ds(self, theta, local_states):
        """
        The function recieves two local states, one refering to the state before the runge-kutta and other refering to a state after runge-kutta and then compute the global state based on the transition
        :param local_states_0: Local state before the transition
        :param local_states_1: Local state after the transition
        :return: global states
        """
        c, s = np.cos(theta), np.sin(theta)
        A = np.array([[c, -s], [s, c]])
        B_l_pos = np.array([local_states[0], local_states[1]])
        B_l_vel = np.array([local_states[3], local_states[4]])

        B_g_pos = np.dot(A, B_l_pos.transpose())
        #print("local",local_states,A,B_l_pos,B_l_vel)
        B_g_vel = np.dot(A, B_l_vel.transpose())

        return np.array([B_g_pos[0], B_g_pos[1], local_states[2], B_g_vel[0], B_g_vel[1], local_states[5]])