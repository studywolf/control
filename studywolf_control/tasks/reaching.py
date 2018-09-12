import controllers.shell as shell
import controllers.forcefield as forcefield

import numpy as np

def Task(arm, controller_class,
        force=None, write_to_file=False, **kwargs):
    """
    This task sets up the arm to move to random
    target positions ever t_target seconds.
    """

    # check controller type ------------------
    controller_name = controller_class.__name__.split('.')[1]
    if controller_name not in ('ddpgmain'):
        raise Exception('Cannot perform reaching task with this controller.')

    # set arm specific parameters ------------
    if arm.DOF == 3:
        kp = 50

    # generate control shell -----------------
    additions = []
    if force is not None:
        print('applying joint velocity based forcefield...')
        additions.append(forcefield.Addition(scale=force))
        task = 'arm%i/forcefield'%arm.DOF

    controller = controller_class.Train(
                                        additions=additions,
                                        kp=kp,
                                        kv=np.sqrt(kp),
                                        task='arm%i/random'%arm.DOF,
                                        write_to_file=write_to_file)
    control_shell = shell.Shell(controller=controller)

    # generate runner parameters -----------
    runner_pars = {'control_type':'random',
                'title':'Task: Random movements'}

    return (control_shell, runner_pars)