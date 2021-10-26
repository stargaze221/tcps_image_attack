#!/usr/bin/env python3
# generates the attack to send to the tello drone

import rospy
from std_msgs.msg import Float32MultiArray        # See https://gist.github.com/jarvisschultz/7a886ed2714fac9f5226
from std_msgs.msg import MultiArrayDimension      # See http://docs.ros.org/api/std_msgs/html/msg/MultiArrayLayout.html
from sensor_msgs.msg import Image
import numpy as np
import torch
from cv_bridge import CvBridge


from setting_params import N_ACT_DIM, N_STATE_DIM, DEVICE, SETTING, FREQ_HIGH_LEVEL
from agents.dynamic_auto_encoder import DynamicAutoEncoderAgent
from agents.rl_agent import DDPGAgent


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
    sub_state_observation = rospy.Subscriber('/airsim_node/state_obs_values', Float32MultiArray, fnc_img_callback1)

    rate=rospy.Rate(FREQ_HIGH_LEVEL)
 

    state_estimator = DynamicAutoEncoderAgent(SETTING, train=True)
    rl_agent = DDPGAgent(SETTING)


    count = 0
    

    while not rospy.is_shutdown():

        '''
        count += 1

        # Load the saved Model every 10 iteration
        if count%10 == 0:
            try:
                state_estimator.load_the_model(777, SETTING['name'])
                rl_agent.load_the_model(777, SETTING['name'])
            except:
                print('An error in loading the saved model. Two possible reasons: 1. no saved model, 2. both of nodes simultaneously try to access the file together')

        if IMAGE_RECEIVED is not None and STATE_OBS_RECEIVED is not None:
             
            ### Update the state estimate ###
            np_im = np.frombuffer(IMAGE_RECEIVED.data, dtype=np.uint8).reshape(IMAGE_RECEIVED.height, IMAGE_RECEIVED.width, -1)
            np_im = np.array(np_im)
            np_state_estimate = state_estimator.step(np_im, prev_np_action).squeeze()

            ### Get action first ###
            prev_torch_state_estimate = torch.FloatTensor(prev_np_state_estimate).to(DEVICE)
            action = rl_agent.get_exploration_action(prev_torch_state_estimate).squeeze()

            ### Calculate the reward ###
            height = STATE_OBS_RECEIVED.layout.dim[0].size
            width = STATE_OBS_RECEIVED.layout.dim[1].size
            np_state_obs_received = np.array(STATE_OBS_RECEIVED.data).reshape((height, width))
            reward = reward1(np_state_obs_received)

            ### State Transition to Pack ###
            # 1. previous state estimate   <-   "prev_np_state_estimate"
            # 2. action                    <-   "action"
            # 3. reward                    <-   "reward"
            # 4. current state estimate    <-   "np_state_estimate"

            np_transition = np.zeros((3, N_STATE_DIM))
            np_transition[0] = prev_np_state_estimate
            np_transition[1][:N_ACT_DIM] = action
            np_transition[1][-1] = reward
            np_transition[2] = np_state_estimate

            msg_mat_transition.layout.dim[0].size = np_transition.shape[0]
            msg_mat_transition.layout.dim[1].size = np_transition.shape[1]
            msg_mat_transition.layout.dim[0].stride = np_transition.shape[0]*np_transition.shape[1]
            msg_mat_transition.layout.dim[1].stride = np_transition.shape[1]
            msg_mat_transition.layout.data_offset = 0
            msg_mat_transition.data = np_transition.flatten().tolist()

            ### Publish the state transition matrix ###
            pub_transition.publish(msg_mat_transition)

            ### Save the current state value.
            prev_np_state_estimate = np_state_estimate
            prev_np_action = action

            torch.cuda.empty_cache()
        '''
            
        rate.sleep()

