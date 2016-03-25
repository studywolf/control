============================================
StudyWolf Control repo
============================================

This is a repository to hold the code I've developed for simulating 
control systems performing different benchmarking tasks. The development 
and theory of the controllers are described at http://studywolf.com

Installation
------------

The control directory requires that you have docopt installed::

   pip install docopt

Additionally, there are a number of arm models available, if you 
wish to use anything other than the 2 link arm coded in python, 
then you will have to compile the arm. You can compile the arms by
going in to the arms/num_link/ folder and running setup::

   python setup.py build_ext -i
   
This will compile the arm for you into a shared object library that's
accessible from Python. 

A final requirement is the pydmps library, which can be installed::

   pip install pydmps

NOTE: The arms have only been tested on linux and currently don't compile on Mac. 

Running
-------

To run the basic control code, from the base directory::

   python run.py ARM CONTROL TASK
   
Where you can find the arm options in the Arm directory subfolders (arm1, arm1_python, arm2, arm2_python, arm2_python_todorov, arm3),the control types available are in the controllers subfolder (gc, osc, lqr, ilqr, dmp, trace), and the tasks are those listed in the Task directory (follow_mouse, postural, random_movements, reach, write).

There are also a bunch of options, browse through the run.py header code to find them all!

If you would like to use the PyGame visualization you must have PyGame installed. To call up the PyGame visualization append --use_pygame=True to the end of your call.
  
Writing to file
---------------

To write to file, run setting the write_to_file flag to True::
  
   python run.py ARM CONTROL TASK --write_to_file=True
  
NOTE: You must create a directory structure in the root folder 'data/arm{2,3}/task_name/controller_name'.
