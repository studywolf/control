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

from Arms.three_link.arm import Arm3Link as Arm3

import Controllers.dmp as dmp 
import Controllers.gc as gc 

from sim_and_plot import Runner
import Tasks.walk as walk

import numpy as np

dt = 1e-3

# instantiate the controller for the walk task
# and get the sim_and_plot parameters 
control_shell, runner_pars = walk.Task()
arm = Arm3(dt=dt)

runner = Runner(dt=dt, **runner_pars)

runner.run(arm=arm, control_shell=control_shell)
runner.show()
