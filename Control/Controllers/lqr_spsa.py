'''
Copyright (C) 2015 Brent Komer and Travis DeWolf

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

import control

import numpy as np
import scipy.linalg as sp_linalg

class Control(control.Control):
    """
    A controller that implements operational space control.
    Controls the (x,y) position of a robotic arm end-effector.
    """
    def __init__(self, solve_continuous=False, adaptation=None, **kwargs): 

        super(Control, self).__init__(**kwargs)

        self.DOF = 2 # task space dimensionality 
        self.u = None
        self.solve_continuous = solve_continuous
        self.adaptation = adaptation

        if self.write_to_file is True:
            from recorder import Recorder
            # set up recorders
            self.u_recorder = Recorder('control signal', self.task, 'lqr')
            self.xy_recorder = Recorder('end-effector position', self.task, 'lqr')
            self.dist_recorder = Recorder('distance from target', self.task, 'lqr')
            self.recorders = [self.u_recorder, 
                            self.xy_recorder, 
                            self.dist_recorder]

    # def calc_derivs(self, x, u):
    #     eps = 0.00001  # finite difference epsilon
    #     #----------- compute xdot_x and xdot_u using finite differences --------
    #     # NOTE: here each different run is in its own column
    #     x1 = np.tile(x, (self.arm.DOF*2,1)).T + np.eye(self.arm.DOF*2) * eps
    #     x2 = np.tile(x, (self.arm.DOF*2,1)).T - np.eye(self.arm.DOF*2) * eps
    #     uu = np.tile(u, (self.arm.DOF*2,1))
    #     f1 = self.plant_dynamics(x1, uu)
    #     f2 = self.plant_dynamics(x2, uu)
    #     xdot_x = (f1 - f2) / 2 / eps
    #
    #     xx = np.tile(x, (self.arm.DOF,1)).T 
    #     u1 = np.tile(u, (self.arm.DOF,1)) + np.eye(self.arm.DOF) * eps
    #     u2 = np.tile(u, (self.arm.DOF,1)) - np.eye(self.arm.DOF) * eps
    #     f1 = self.plant_dynamics(xx, u1)
    #     f2 = self.plant_dynamics(xx, u2)
    #     xdot_u = (f1 - f2) / 2 / eps
    #
    #     return xdot_x, xdot_u

    def calc_derivs(self, x, u, a=.101, A=.193, c=.0277):
        """ calculate gradient of plant dynamics using Simultaneous
        Perturbation Stochastic Approximation (SPSA). Implemented 
        base on (Spall, 1998).

        x np.array: the state of the system
        u np.array: the control signal 
        """  
        # Step 1: Initialization and coefficient selection
        k = 0 # iteration counter
        max_iters = 4
        converge_thresh = 1e-10

        alpha = 0.602 # from (Spall, 1998)
        gamma = 0.101
        a = 0.101 # found using hyperopt
        A = .193
        c = .0277

        delta_K = None
        delta_J = None
        for ii in range(max_iters):
            ak = a / (A + k + 1)**alpha
            ck = c / (k + 1)**gamma

            # Step 2: Generation of simultaneous perturbation vector
            # choose each component from a bernoulli +-1 distribution with 
            # probability of .5 for each +-1 outcome.
            delta_k = np.random.choice([-1,1], size=len(x) + len(u), p=[.5, .5])

            # Step 3: Function evaluations
            inc_x = np.copy(x) + ck * delta_k[:len(x)]
            inc_u = np.copy(u) + ck * delta_k[len(x):]
            state_inc = self.plant_dynamics(inc_x, inc_u)
            dec_x = np.copy(x) - ck * delta_k[:len(x)]
            dec_u = np.copy(u) - ck * delta_k[len(x):]
            state_dec = self.plant_dynamics(dec_x, dec_u)

            delta_j = ((state_inc - state_dec) / (2.0*ck)).reshape(-1)

            # Step 4: Track delta_k and delta_j for Gradient approximation
            delta_K = delta_k if delta_K is None else \
                np.vstack([delta_K, delta_k])
            delta_J =  delta_j if delta_J is None else \
                np.vstack([delta_J, delta_j])

        f_xu = np.dot(np.linalg.pinv(np.dot(delta_K.T, delta_K)), 
                np.dot(delta_K.T, delta_J))
        f_x = f_xu[:len(x)]
        f_u = f_xu[len(x):]

        return f_x.T , f_u.T

    def control(self, arm, x_des=None):
        """Generates a control signal to move the 
        arm to the specified target.
            
        arm Arm: the arm model being controlled
        des list: the desired system position
        x_des np.array: desired task-space force, 
                        system goes to self.target if None
        """
        if self.u is None:
            self.u = np.zeros(arm.DOF)

        self.Q = np.zeros((arm.DOF*2, arm.DOF*2))
        self.Q[:arm.DOF, :arm.DOF] = np.eye(arm.DOF) * 1000.0 
        self.R = np.eye(arm.DOF) * 0.001 

        # calculate desired end-effector acceleration
        if x_des is None:
            self.x = arm.x 
            x_des = self.x - self.target 

        self.arm, state = self.copy_arm(arm)
        A, B = self.calc_derivs(state, self.u)

        if self.solve_continuous is True:
            X = sp_linalg.solve_continuous_are(A, B, self.Q, self.R)
            K = np.dot(np.linalg.pinv(self.R), np.dot(B.T, X))
        else: 
            X = sp_linalg.solve_discrete_are(A, B, self.Q, self.R)
            K = np.dot(np.linalg.pinv(self.R + np.dot(B.T, np.dot(X, B))), np.dot(B.T, np.dot(X, A)))

        # transform the command from end-effector space to joint space
        J = self.arm.gen_jacEE()

        u = np.hstack([np.dot(J.T, x_des), arm.dq])

        self.u = -np.dot(K, u)

        if self.write_to_file is True:
            # feed recorders their signals
            self.u_recorder.record(0.0, self.u)
            self.xy_recorder.record(0.0, self.x)
            self.dist_recorder.record(0.0, self.target - self.x)

        # add in any additional signals 
        for addition in self.additions:
            self.u += addition.generate(self.u, arm)

        return self.u
 
    def copy_arm(self, real_arm):

        # need to make a copy of the arm for simulation
        arm = real_arm.__class__()
        arm.dt = real_arm.dt

        # reset arm position to x_0
        arm.reset(q = real_arm.q, dq = real_arm.dq)

        return arm, np.hstack([real_arm.q, real_arm.dq])

    def plant_dynamics(self, x, u):

        if x.ndim == 1:
            x = x[:,None]
            u = u[None,:]

        xnext = np.zeros((x.shape))
        for ii in range(x.shape[1]):
            # set the arm position to x
            self.arm.reset(q=x[:self.arm.DOF, ii], 
                          dq=x[self.arm.DOF:self.arm.DOF*2, ii])

            # apply the control signal
            # TODO: should we be using a constant timestep here instead of arm.dt?
            # to even things out when running at different dts? 
            self.arm.apply_torque(u[ii], self.arm.dt)
            # get the system state from the arm
            xnext[:,ii] = np.hstack([np.copy(self.arm.q), 
                                   np.copy(self.arm.dq)])

        if self.solve_continuous is True:
            xdot = ((np.asarray(xnext) - np.asarray(x)) / self.arm.dt).squeeze()
            return xdot
        return xnext

    def gen_target(self, arm):
        gain = np.sum(arm.L) * .75
        bias = -np.sum(arm.L) * 0
        
        self.target = np.random.random(size=(2,)) * gain + bias

        return self.target.tolist()
