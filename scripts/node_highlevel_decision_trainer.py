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

    pub_loss_monitor = rospy.Publisher('/decision_trainer_node/loss_monitor', Float32MultiArray, queue_size=1)   # publisher1 initialization.

    rate=rospy.Rate(FREQ_HIGH_LEVEL)
 
    state_estimator = DynamicAutoEncoderAgent(SETTING, train=True)
    rl_agent = DDPGAgent(SETTING)

    single_trajectory_memory = SingleTrajectoryBuffer(1000)
    transition_memory = TransitionBuffer(1000)

    n_iteration = 0

    # msg init. the msg is to send out numpy array.
    msg_mat = Float32MultiArray()
    msg_mat.layout.dim.append(MultiArrayDimension())
    msg_mat.layout.dim.append(MultiArrayDimension())
    msg_mat.layout.dim[0].label = "height"
    msg_mat.layout.dim[1].label = "width"

    N_LOG_INTERVAL = 10
    

    while not rospy.is_shutdown():

        n_iteration += 1
        

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
            done = np_transition[1][-2]
            np_state_estimate = np_transition[2]

            single_trajectory_memory.add(np_im, action, prev_np_state_estimate)
            transition_memory.add(prev_np_state_estimate, action, reward, np_state_estimate, done)

            ### Update ###
            if single_trajectory_memory.len > N_WINDOW and transition_memory.len > N_WINDOW:

                batch_obs_img_stream, batch_tgt_stream, batch_state_est_stream = single_trajectory_memory.sample(N_WINDOW)
                loss_sys_id = state_estimator.update(batch_obs_img_stream, batch_state_est_stream, batch_tgt_stream)

                s_arr, a_arr, r_arr, s1_arr, done_arr = transition_memory.sample(8)
                loss_actor, loss_critic = rl_agent.update(s_arr, a_arr, r_arr, s1_arr, done_arr)

                loss_monitor_np = np.array([[loss_sys_id, loss_actor, loss_critic]])

                msg_mat.layout.dim[0].size = loss_monitor_np.shape[0]
                msg_mat.layout.dim[1].size = loss_monitor_np.shape[1]
                msg_mat.layout.dim[0].stride = loss_monitor_np.shape[0]*loss_monitor_np.shape[1]
                msg_mat.layout.dim[1].stride = loss_monitor_np.shape[1]
                msg_mat.layout.data_offset = 0
                msg_mat.data = loss_monitor_np.flatten().tolist()
                pub_loss_monitor.publish(msg_mat)

            torch.cuda.empty_cache()

        if n_iteration % N_LOG_INTERVAL==0:
            try:
                state_estimator.save_the_model()
                rl_agent.save_the_model()
            except:
                print('in high_level_decision_trainer, model saving failed!')

            
            
        rate.sleep()

