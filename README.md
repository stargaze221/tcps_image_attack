# TCPS_Image_Attack

AirSim does not work well with Ubuntu 20.

We had to use ROS Melodic that works with Ubuntu 18.

And Default ROS Melodic does not work with Python2.

So, we used the scheme linked below. It is basically reinstall some ROS relevant packages.


https://dhanoopbhaskar.com/blog/2020-05-07-working-with-python-3-in-ros-kinetic-or-melodic/


ROS (upto Melodic) officially supports only python2 and NOT python3. However some libraries we use in our projects (eg. Speech Recognition using Google Cloud Speech) may require python3 to run.

If ROS needs to support python3 we may have to recompile ROS source code using python3 which is not practical.

So what we can do is to run python3 programs, separately and connect using ROS bridge. (if we use custom messages (ROS msg)

However, if we are not using any custom rosmsg and using only built-in rosmsg, we can do the following steps to run python3 codes in ROS (without using a ROS bridge.)

Install ROS (here I install Melodic)

 apt install ros-melodic-desktop-full

After installing ROS, install rospkg for python3

 apt install python3-pip python3-all-dev python3-rospkg

This will prompt to install python3-rospkg and to remove ROS packages (already installed). Select Yes for that prompt. This will remove ROS packages and we will have to re-install them.

 apt install ros-melodic-desktop-full --fix-missing

This will complete the installation part. Now comes the coding part.

Just include the following directive as the first line of your program code (file) which should be executed using python3.

 #!/usr/bin/env python3

We can now execute everything as we do normally in ROS. Read the documentation (link is given above) for more information on ROS.