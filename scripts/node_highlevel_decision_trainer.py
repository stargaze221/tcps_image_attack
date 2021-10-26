#!/usr/bin/env python3
# generates the attack to send to the tello drone

import rospy
from std_msgs.msg import Float32MultiArray        # See https://gist.github.com/jarvisschultz/7a886ed2714fac9f5226
from std_msgs.msg import MultiArrayDimension      # See http://docs.ros.org/api/std_msgs/html/msg/MultiArrayLayout.html
from sensor_msgs.msg import Image
import numpy as np
import torch
import cv2

from setting_params import N_ACT_DIM, N_STATE_DIM, DEVICE, SETTING, FREQ_HIGH_LEVEL, N_WINDOW
from agents.dynamic_auto_encoder import DynamicAutoEncoderAgent
from agents.rl_agent import DDPGAgent


from memory import SingleTrajectoryBuffer, TransitionBuffer 

IMAGE_RECEIVED = None
def fnc_img_callback(msg):
    global IMAGE_RECEIVED
    IMAGE_RECEIVED = msg

TRANSITION_EST_RECEIVED = None
def fnc_img_callback1(msg):
    global TRANSITION_EST_RECEIVED
    TRANSITION_EST_RECEIVED = msg



if __name__ == '__main__':
    '''
    Input: image frame
    Outputs: previous state, action, reward, state_estimate
    '''
    rospy.init_node('high_level_decision_trainer')
    sub_image = rospy.Subscriber('/airsim_node/camera_frame', Image, fnc_img_callback)
    sub_state_observation = rospy.Subscriber('/decision_maker_node/state_est_transition', Float32MultiArray, fnc_img_callback1)

    rate=rospy.Rate(FREQ_HIGH_LEVEL)
 
    state_estimator = DynamicAutoEncoderAgent(SETTING, train=True)
    rl_agent = DDPGAgent(SETTING)

    single_trajectory_memory = SingleTrajectoryBuffer(1000)
    transition_memory = TransitionBuffer(1000)

    count = 0
    

    while not rospy.is_shutdown():

        count += 1

        # Load the saved Model every 10 iteration
        if count%10 == 0:
            try:
                state_estimator.load_the_model(777, SETTING['name'])
                rl_agent.load_the_model(777, SETTING['name'])
            except:
                print('An error in loading the saved model. Two possible reasons: 1. no saved model, 2. both of nodes simultaneously try to access the file together')

        if IMAGE_RECEIVED is not None and TRANSITION_EST_RECEIVED is not None:

            ### Add samples to the buffers ###
            np_im = np.frombuffer(IMAGE_RECEIVED.data, dtype=np.uint8).reshape(IMAGE_RECEIVED.height, IMAGE_RECEIVED.width, -1)
            np_im = np.array(np_im)
            np_im = cv2.resize(np_im, SETTING['encoder_image_size'], interpolation = cv2.INTER_AREA)

            height = TRANSITION_EST_RECEIVED.layout.dim[0].size
            width = TRANSITION_EST_RECEIVED.layout.dim[1].size
            np_transition = np.array(TRANSITION_EST_RECEIVED.data).reshape((height, width))

            prev_np_state_estimate = np_transition[0]
            action = np_transition[1][:N_ACT_DIM]
            reward = np_transition[1][-1]
            np_state_estimate = np_transition[2]

            single_trajectory_memory.add(np_im, action, prev_np_state_estimate)
            transition_memory.add(prev_np_state_estimate, action, reward, np_state_estimate)

            ### Update ###
            if single_trajectory_memory.len > N_WINDOW:
                batch_obs_img_stream, batch_tgt_stream, batch_state_est_stream = single_trajectory_memory.sample(N_WINDOW)
                loss_sys_id = state_estimator.update(batch_obs_img_stream, batch_state_est_stream, batch_tgt_stream)

            if transition_memory.len > N_WINDOW:
                s_arr, a_arr, r_arr, s1_arr = transition_memory.sample(8)
                loss_actor, loss_critic_= rl_agent.update(s_arr, a_arr, r_arr, s1_arr)

            torch.cuda.empty_cache()
            
        rate.sleep()

