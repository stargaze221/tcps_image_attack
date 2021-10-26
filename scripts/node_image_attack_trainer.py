#!/usr/bin/env python3
import rospy
import torch
import numpy as np
import PIL
import os

from std_msgs.msg import Float32
from std_msgs.msg import Float32MultiArray        # See https://gist.github.com/jarvisschultz/7a886ed2714fac9f5226
from std_msgs.msg import MultiArrayDimension      # See http://docs.ros.org/api/std_msgs/html/msg/MultiArrayLayout.html
from geometry_msgs.msg import Twist

from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from memory import ImageBuffer, ImageTargetBuffer
from model import ImageAttackNetwork

from agents.image_attack_agent import ImageAttackTraniner

from setting_params import WRITER, DEVICE, FREQ_HIGH_LEVEL


N_MEMORY_SIZE = 1000
N_MINIBATCH = 3

IMAGE_TGT_MEMORY = ImageTargetBuffer(N_MEMORY_SIZE)
IMAGEATTACKNN = ImageAttackNetwork(h=448, w=448, action_dim=4).to(DEVICE)

FREQ_NODE = FREQ_HIGH_LEVEL
N_LOG_INTERVAL = 5

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
    rospy.init_node('image_attack_train_node')   # rosnode node initialization
    print('Image_attack_train_node is initialized at', os.getcwd())

    sub_image = rospy.Subscriber('/airsim_node/camera_frame', Image, fnc_img_callback)   # subscriber init.
    sub_target = rospy.Subscriber('/decision_maker_node/target', Twist, fnc_target_callback)

    rate=rospy.Rate(FREQ_NODE)   # Running rate at 20 Hz
    agent = ImageAttackTraniner()

    n_iteration = 0
    ##############################
    ### Instructions in a loop ###
    ##############################
    while not rospy.is_shutdown():

        if IMAGE_RECEIVED is not None and TARGET_RECEIVED is not None:
            # Add data into memory
            np_im = np.frombuffer(IMAGE_RECEIVED.data, dtype=np.uint8).reshape(IMAGE_RECEIVED.height, IMAGE_RECEIVED.width, -1)
            act = np.array([TARGET_RECEIVED.linear.x, TARGET_RECEIVED.linear.y, TARGET_RECEIVED.linear.z, TARGET_RECEIVED.angular.x])
            #act = np.array([-0.5, -0.5, -0.5, -0.5])  # or np.random.rand(4)
            IMAGE_TGT_MEMORY.add(np_im, act)

            # Sample data from the memory
            minibatch_img, minibatch_act = IMAGE_TGT_MEMORY.sample(N_MINIBATCH) # list of numpy arrays
            minibatch_img = np.array(minibatch_img).astype(np.float32) # cast it into a numpy array
            
            ####################################################
            ## CAL THE LOSS FUNCTION & A STEP OF GRAD DESCENT ##
            ####################################################
            agent.update(minibatch_img, minibatch_act)
            torch.cuda.empty_cache()


            
            n_iteration += 1

            # # Log the loss values
            # dict_sum_loss['sum_loss_attack'] = dict_sum_loss['sum_loss_attack'] + dict_loss_values['loss_attack'] 
            # dict_sum_loss['sum_error_confidence'] = dict_sum_loss['sum_error_confidence'] + dict_loss_values['error_obj_confidence'] 
            

            # dict_sum_loss['n_count'] = dict_sum_loss['n_count'] + 1 

            if n_iteration % N_LOG_INTERVAL==0:
                #print(os.getcwd())
                agent.save_the_model(777)

            #     WRITER.add_scalar('imageattack_loss', dict_sum_loss['sum_loss_attack']/dict_sum_loss['n_count'], n_iteration)
            #     WRITER.add_scalar('error_obj_confidence', dict_sum_loss['sum_error_confidence']/dict_sum_loss['n_count'], n_iteration)
            #     dict_sum_loss = {'sum_loss_attack':0, 'sum_error_confidence':0, 'sum_l2_loss':0, 'n_count':0}
            #     agent.save_the_model(777)

            



        rate.sleep()

        #break
