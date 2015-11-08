'''
Copyright (C) 2015 Travis DeWolf and Brent Komer

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
import scipy.linalg as spla

from Arms.one_link.arm_python import Arm1Link as Arm1_python
from Arms.one_link.arm import Arm1Link as Arm1
from Arms.two_link.arm_python import Arm2Link as Arm2_python
from Arms.two_link.arm import Arm2Link as Arm2
from Arms.three_link.arm import Arm3Link as Arm3

class Control(control.Control):
    """
    A controller that implements operational space control.
    Controls the (x,y) position of a robotic arm end-effector.
    """
    def __init__(self, solve_continuous=False, **kwargs): 

        super(Control, self).__init__(**kwargs)

        self.DOF = 2 # task space dimensionality 
        self.u = None
        self.solve_continuous = solve_continuous

    def calc_derivs(self, x, u):
        eps = 0.00001  # finite difference epsilon
        #----------- compute xdot_x and xdot_u using finite differences --------
        # NOTE: here each different run is in its own column
        x1 = np.tile(x, (self.arm.DOF*2,1)).T + np.eye(self.arm.DOF*2) * eps
        x2 = np.tile(x, (self.arm.DOF*2,1)).T - np.eye(self.arm.DOF*2) * eps
        uu = np.tile(u, (self.arm.DOF*2,1))
        # need xdot, so subtract xx, since fn returns x(k+1)
        f1 = self.fn_dyn(x1, uu)
        f2 = self.fn_dyn(x2, uu)
        xdot_x = (f1 - f2) / 2 / eps
   
        xx = np.tile(x, (self.arm.DOF,1)).T 
        u1 = np.tile(u, (self.arm.DOF,1)) + np.eye(self.arm.DOF) * eps
        u2 = np.tile(u, (self.arm.DOF,1)) - np.eye(self.arm.DOF) * eps
        # need xdot, so subtract xx, since fn returns x(k+1)
        f1 = self.fn_dyn(xx, u1)
        f2 = self.fn_dyn(xx, u2)
        xdot_u = (f1 - f2) / 2 / eps

        return xdot_x, xdot_u

    def check_distance(self, arm):
        """Checks the distance to target"""
        return np.sum(abs(arm.x - self.target))

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
            self.x = arm.position(ee_only=True)
            x_des = self.x - self.target 

        state = np.hstack([arm.q, arm.dq]) 
        self.arm = arm.copy()
        A, B = self.calc_derivs(state, self.u)

        if self.solve_continuous is True:
            X = spla.solve_continuous_are(A, B, self.Q, self.R)
            K = np.dot(np.linalg.pinv(self.R), np.dot(B.T, X))
        else: 
            X = spla.solve_discrete_are(A, B, self.Q, self.R)
            K = np.dot(np.linalg.pinv(self.R + np.dot(B.T, np.dot(X, B))), np.dot(B.T, np.dot(X, A)))

        # transform the command from end-effector space to joint space
        J = arm.gen_jacEE()
        u = np.hstack([np.dot(J.T, x_des), arm.dq])

        self.u = -np.dot(K, u)

        return self.u
 
    def fn_dyn(self, x, u):

        if x.ndim == 1:
            x = x[:,None]
            u = u[None,:]

        xnext = np.zeros((x.shape))
        for ii in range(x.shape[1]):
            # set the arm position to x
            self.arm.reset(q=x[:self.arm.DOF, ii], 
                          dq=x[self.arm.DOF:self.arm.DOF*2, ii])

            # apply the control signal
            self.arm.apply_torque(u[ii], self.arm.dt)
            # get the system state from the arm
            xnext[:,ii] = np.hstack([np.copy(self.arm.q), 
                                   np.copy(self.arm.dq)])

        if self.solve_continuous is True:
            xdot = ((np.asarray(xnext) - np.asarray(x)) / self.arm.dt).squeeze()
            return xdot
        return xnext

    def gen_target(self, arm):
        """Generate a random target"""
        gain = np.sum(arm.L) * 1.5
        bias = -np.sum(arm.L) * .75
        
        self.target = np.random.random(size=(2,)) * gain + bias

        return self.target.tolist()
