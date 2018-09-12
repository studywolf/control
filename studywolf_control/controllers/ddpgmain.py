from . import control
import numpy as np
from . import ddpg

class Train(control.Control):
    """
    A controller that implements operational space control.
    Controls the (x,y) position of a robotic arm end-effector.
    """
    def __init__(self, **kwargs):
        """
        null_control boolean: apply second controller in null space or not
        """

        super(Train, self).__init__(**kwargs)

        self.DOF = 2 # task space dimensionality

        if self.write_to_file is True:
            from recorder import Recorder
            # set up recorders
            self.u_recorder = Recorder('control signal', self.task, 'ddpgmain')
            self.xy_recorder = Recorder('end-effector position', self.task, 'ddpgmain')
            self.dist_recorder = Recorder('distance from target', self.task, 'ddpgmain')
            self.recorders = [self.u_recorder,
                            self.xy_recorder,
                            self.dist_recorder]

    def control(self, arm, x_des=None):
        """Generates a control signal to move the
        arm to the specified target.

        arm Arm: the arm model being controlled
        des list: the desired system position
        x_des np.array: desired task-space force,
                        system goes to self.target if None
        """
        # calculate desired end-effector acceleration
        if x_des is None:
            self.x = arm.x
            x_des = self.kp * (self.target - self.x)

        # generate the mass matrix in end-effector space
        Mq = arm.gen_Mq()
        Mx = arm.gen_Mx()

        # calculate force
        Fx = np.dot(Mx, x_des)

        # calculate the Jacobian
        JEE = arm.gen_jacEE()
        # tau = J^T * Fx + tau_grav, but gravity = 0
        # add in velocity compensation in GC space for stability
        self.u = (np.dot(JEE.T, Fx).reshape(-1,) -
                  np.dot(Mq, self.kv * arm.dq))

        if self.write_to_file is True:
            # feed recorders their signals
            self.u_recorder.record(0.0, self.u)
            self.xy_recorder.record(0.0, self.x)
            self.dist_recorder.record(0.0, self.target - self.x)

        # add in any additional signals
        #for addition in self.additions:
            #self.u += addition.generate(self.u, arm)

        return self.u

    def gen_target(self, arm):
        """Generate a random target"""
        gain = np.sum(arm.L) * .75
        bias = -np.sum(arm.L) * 0

        self.target = np.random.random(size=(2,)) * gain + bias
        #self.target = np.array([1, 1])

        return self.target.tolist()
