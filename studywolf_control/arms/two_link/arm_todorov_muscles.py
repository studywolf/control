'''
Copyright (C) 2015 Travis DeWolf

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

from arm2base import Arm2Base
import numpy as np

class Arm(Arm2Base):
    """
    A class that holds the simulation and control dynamics for 
    a two link arm, with the dynamics carried out in Python.
    This implementation is based off of the model described in 
    'Iterative linear quadratic regulator design for nonlinear
    biological movement systems' by Li and Torodov
    """
    def __init__(self, **kwargs): 
        
        Arm2Base.__init__(self, **kwargs)

        self.M = np.zeros((2,6)) # moment arms of muscles
        self.a = np.zeros((1,6)) # muscle activations

        self.reset() # set to init_q and init_dq

    def apply_torque(self, u, dt=None):
        dt = self.dt if dt is None else dt
    
        # activation of muscles is neural input u put through
        # a filter describing the calcium dynamics
        t_act = .05
        t_deact = .066 
        t = t_deact + u * (t_act - t_deact) 
        t[u <= a] = t_deact
        self.a += (u - self.a) / t(u, a) * dt

        # muscles moment arm [[shoulder, elbow]]
        # perpendicular distance from the muscle's line of 
        # action to the joint's center of rotation
        self.M[0] = np.array([0, a + b * np.cos(c * q[1])]) # elbow flexor
        self.M[1] = np.array([0, -1.9]) # elbow extensor - ish, should get actual number
        self.M[2] = np.array([a + b * np.cos(c * q[0]), 0]) # shoulder flexor
        self.M[3] = np.array([-4.5, 0]) # shoulder extensor - ish, should get actual number
        self.M[4] = np.array([a + b * np.cos(c * q[0]), # biarticulate flexor
                              a + b * np.cos(c * q[1])])
        self.M[5] = np.array([-3.9, -2.5]) # biarticulate extensor - ish, should get actual numbers

        # muscle length and velocity are calculated as 
        # functions of joint angle and angular velocity

        # given the activation signal, and muscle lengths and velocities
        # we can now calculate the joint torques (tau) applied to the arm
        FP = -0.02 * np.exp(13.8 - 18.7 * l)
        if v < 0.0: 
            FV =  (-0.572 - v) / (-5.72 + (1.38 + 2.09 * l) * v)
        else:
            FV = 0.62 - (-3.12 + 4.21 * l - 2.67 * l**2) * v / 0.62 + v
        FL = np.exp(-np.abs((l**1.93 - 1.0) / 1.03)**1.87)
        Nf = 2.11 + 4.16 * (1.0 / l - 1.0)
        A = 1.0 - np.exp(-(a / (0.56 * Nf)**Nf))
        T = A*(FL * FV + FP)
        tau = self.M * T

        # arm model parameters
        m = np.array([1.4, 1.1]) # segment mass
        l = np.array([0.3, 0.33]) # segment length
        s = np.array([0.11, 0.16]) # segment center of mass
        i = np.array([0.025, 0.045]) # segment moment of inertia

        # inertia matrix
        a1 = i[0] + i[1] + m[1]*l[0]**2
        a2 = m[1]*l[0]*s[1]
        a3 = i[1]
        I = np.array([[a1 + 2*a2*np.cos(q[1]), a3 + a2*np.cos(q[1])],
                      [a3 + a2*np.cos(q[1]), a3]])

        # centripital and Coriolis effects
        C = np.array([[-dq[1] * (2 * dq[0] + dq[1])],
                      [dq[0]]]) * a2 * np.sin(q[1])

        # joint friction
        B = np.array([[.05, .025],
                      [.025, .05]])

        # calculate forward dynamics
        ddq = np.linalg.pinv(I) * (tau - C - np.dot(B, dq))

        # transfer to next time step 
        self.q += dt * self.dq
        self.dq += dt * ddq

        self.t += dt
