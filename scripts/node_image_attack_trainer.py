#!/usr/bin/env python3
import rospy
import torch
import numpy as np
import os

from std_msgs.msg import Float32MultiArray        # See https://gist.github.com/jarvisschultz/7a886ed2714fac9f5226
from std_msgs.msg import MultiArrayDimension      # See http://docs.ros.org/api/std_msgs/html/msg/MultiArrayLayout.html
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image

from memory import ImageBuffer, ImageTargetBuffer
from agents.image_attack_agent import ImageAttackTraniner

from setting_params import SETTING, DEVICE, FREQ_HIGH_LEVEL

IMAGE_TGT_MEMORY = ImageTargetBuffer(SETTING['N_ImageBuffer'])

### ROS Subscriber Callback ###
IMAGE_RECEIVED = None
def fnc_img_callback(msg):
    global IMAGE_RECEIVED
    IMAGE_RECEIVED = msg

TARGET_RECEIVED = None
def fnc_target_callback(msg):
    global TARGET_RECEIVED
    TARGET_RECEIVED = msg

if __name__=='__main__':

    # rosnode node initialization
    rospy.init_node('image_attack_train_node')   # rosnode node initialization
    print('Image_attack_train_node is initialized at', os.getcwd())

    # subscriber init.
    sub_image = rospy.Subscriber('/airsim_node/camera_frame', Image, fnc_img_callback)   # subscriber init.
    sub_target = rospy.Subscriber('/decision_maker_node/target', Twist, fnc_target_callback)

    # publishers init.
    pub_loss_monitor = rospy.Publisher('/image_attack_train_node/loss_monitor', Float32MultiArray, queue_size=3)   # publisher1 initialization.

    # Running rate
    rate=rospy.Rate(FREQ_HIGH_LEVEL)

    # Training agents init
    SETTING['name'] = rospy.get_param('name')
    agent = ImageAttackTraniner(SETTING)

    # msg init. the msg is to send out numpy array.
    msg_mat = Float32MultiArray()
    msg_mat.layout.dim.append(MultiArrayDimension())
    msg_mat.layout.dim.append(MultiArrayDimension())
    msg_mat.layout.dim[0].label = "height"
    msg_mat.layout.dim[1].label = "width"

    n_iteration = 0
    ##############################
    ### Instructions in a loop ###
    ##############################
    while not rospy.is_shutdown():

        if IMAGE_RECEIVED is not None and TARGET_RECEIVED is not None:
            n_iteration += 1

            # Add data into memory
            np_im = np.frombuffer(IMAGE_RECEIVED.data, dtype=np.uint8).reshape(IMAGE_RECEIVED.height, IMAGE_RECEIVED.width, -1)
            act = np.array([TARGET_RECEIVED.linear.x, TARGET_RECEIVED.linear.y, TARGET_RECEIVED.linear.z, TARGET_RECEIVED.angular.x])
            #act = np.array([-0.5, -0.5, -0.5, -0.5])  # or np.random.rand(4)
            IMAGE_TGT_MEMORY.add(np_im, act)

            # Sample data from the memory
            minibatch_img, minibatch_act = IMAGE_TGT_MEMORY.sample(SETTING['N_MINIBATCH_IMG']) # list of numpy arrays
            minibatch_img = np.array(minibatch_img).astype(np.float32) # cast it into a numpy array
            
            ####################################################
            ## CAL THE LOSS FUNCTION & A STEP OF GRAD DESCENT ##
            ####################################################
            loss_value = agent.update(minibatch_img, minibatch_act)
            loss_monitor_np = np.array([[loss_value]])
            msg_mat.layout.dim[0].size = loss_monitor_np.shape[0]
            msg_mat.layout.dim[1].size = loss_monitor_np.shape[1]
            msg_mat.layout.dim[0].stride = loss_monitor_np.shape[0]*loss_monitor_np.shape[1]
            msg_mat.layout.dim[1].stride = loss_monitor_np.shape[1]
            msg_mat.layout.data_offset = 0
            msg_mat.data = loss_monitor_np.flatten().tolist()
            pub_loss_monitor.publish(msg_mat)

            if n_iteration % (FREQ_HIGH_LEVEL+1)==0:
                try:
                    agent.save_the_model()
                except:
                    print('In image_attack_train_node, model saving failed!')

        try:
            experiment_done_done = rospy.get_param('experiment_done')
        except:
            experiment_done_done = False
        if experiment_done_done and n_iteration > FREQ_HIGH_LEVEL*3:
            rospy.signal_shutdown('Finished 100 Episodes!')

                    
        rate.sleep()
