#!/usr/bin/env python3

# Train the state estimator (dynamics autoencoder) and the RL agent that generates target (or action)

import rospy
from std_msgs.msg import Float32MultiArray        # See https://gist.github.com/jarvisschultz/7a886ed2714fac9f5226
from std_msgs.msg import MultiArrayDimension      # See http://docs.ros.org/api/std_msgs/html/msg/MultiArrayLayout.html
from sensor_msgs.msg import Image
import numpy as np
import cv2

from setting_params import SETTING, FREQ_HIGH_LEVEL, DEVICE
from agents.image_rl_agent import ImageDDPGAgent

from memory import TransitionBuffer


TRANSITION_IM1 = None
def fnc_img_callback1(msg):
    global TRANSITION_IM1
    TRANSITION_IM1 = msg

TRANSITION_A1_R1 = None
def fnc_img_callback2(msg):
    global TRANSITION_A1_R1
    TRANSITION_A1_R1 = msg

TRANSITION_IM2 = None
def fnc_img_callback3(msg):
    global TRANSITION_IM2
    TRANSITION_IM2 = msg



if __name__ == '__main__':
    
    # rosnode node initialization
    rospy.init_node('high_level_decision_trainer_wo_dynenc')

    # subscriber init.
    sub_transition_im1 = rospy.Subscriber('/decision_maker_node/transition_im1', Image, fnc_img_callback1)
    sub_transition_a1_r1 = rospy.Subscriber('/decision_maker_node/transition_a1_r1', Float32MultiArray, fnc_img_callback2)
    sub_transition_im1 = rospy.Subscriber('/decision_maker_node/transition_im2', Image, fnc_img_callback3)

    # publishers init.
    pub_loss_monitor = rospy.Publisher('/decision_trainer_node/loss_monitor', Float32MultiArray, queue_size=3)   # publisher1 initialization.

    # Running rate
    rate=rospy.Rate(FREQ_HIGH_LEVEL)

    # Training agents init
    SETTING['name'] = rospy.get_param('name')
    rl_agent = ImageDDPGAgent(SETTING)

    # Memory init
    transition_memory = TransitionBuffer(SETTING['N_TransitionBuffer'])

    # msg init. the msg is to send out numpy array.
    msg_mat = Float32MultiArray()
    msg_mat.layout.dim.append(MultiArrayDimension())
    msg_mat.layout.dim.append(MultiArrayDimension())
    msg_mat.layout.dim[0].label = "height"
    msg_mat.layout.dim[1].label = "width"

    ##############################
    ### Instructions in a loop ###
    ##############################
    n_iteration = 0
    while not rospy.is_shutdown():

        if TRANSITION_IM1 is not None and TRANSITION_IM2 is not None and TRANSITION_A1_R1 is not None:
            n_iteration += 1

            ### Add samples to the buffers ###
            # unpack image
            np_im1 = np.frombuffer(TRANSITION_IM1.data, dtype=np.uint8).reshape(TRANSITION_IM1.height, TRANSITION_IM1.width, -1)
            np_im2 = np.frombuffer(TRANSITION_IM2.data, dtype=np.uint8).reshape(TRANSITION_IM2.height, TRANSITION_IM2.width, -1)

            # unpack state
            height = TRANSITION_A1_R1.layout.dim[0].size
            width = TRANSITION_A1_R1.layout.dim[1].size
            np_transition = np.array(TRANSITION_A1_R1.data).reshape((height, width))

            # pack state transition
            action = np_transition[0][:SETTING['N_ACT_DIM']]
            reward = np_transition[0][-2]
            done = np_transition[0][-1]

            # add data into memory
            transition_memory.add(np_im1, action, reward, np_im2, done)

            ####################################################
            ## CAL THE LOSS FUNCTION & A STEP OF GRAD DESCENT ##
            ####################################################
            if transition_memory.len > SETTING['N_WINDOW']:

                try:

                    # sample minibach
                    s_arr, a_arr, r_arr, s1_arr, done_arr = transition_memory.sample(SETTING['N_MINIBATCH_DDPG'])

                    # update the models
                    loss_sys_id = 0
                    loss_actor, loss_critic = rl_agent.update(s_arr, a_arr, r_arr, s1_arr, done_arr)

                    # if n_iteration > 2000:
                    #     loss_actor, loss_critic = rl_agent.update(s_arr, a_arr, r_arr, s1_arr, done_arr)
                    # else:
                    #     loss_actor = 0
                    #     loss_critic = 0

                    # pack up loss values
                    loss_monitor_np = np.array([[loss_sys_id, loss_actor, loss_critic]])
                    msg_mat.layout.dim[0].size = loss_monitor_np.shape[0]
                    msg_mat.layout.dim[1].size = loss_monitor_np.shape[1]
                    msg_mat.layout.dim[0].stride = loss_monitor_np.shape[0]*loss_monitor_np.shape[1]
                    msg_mat.layout.dim[1].stride = loss_monitor_np.shape[1]
                    msg_mat.layout.data_offset = 0
                    msg_mat.data = loss_monitor_np.flatten().tolist()
                    pub_loss_monitor.publish(msg_mat)

                except:
                    print('error during the udpate!')

        if n_iteration % (FREQ_HIGH_LEVEL+1) ==0:
            try:
                rl_agent.save_the_model()
            except:
                print('in high_level_decision_trainer, model saving failed!')
        
        try:
            experiment_done_done = rospy.get_param('experiment_done')
        except:
            experiment_done_done = False
        if experiment_done_done and n_iteration > FREQ_HIGH_LEVEL*3:
            rospy.signal_shutdown('Finished 100 Episodes!')
            
            
        rate.sleep()

        

