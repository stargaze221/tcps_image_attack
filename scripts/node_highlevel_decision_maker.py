#!/usr/bin/env python3
# generates the attack to send to the tello drone

import rospy
from std_msgs.msg import Float32MultiArray        # See https://gist.github.com/jarvisschultz/7a886ed2714fac9f5226
from std_msgs.msg import MultiArrayDimension      # See http://docs.ros.org/api/std_msgs/html/msg/MultiArrayLayout.html
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
import numpy as np
import torch
import cv2


from setting_params import N_ACT_DIM, N_STATE_DIM, DEVICE, SETTING, FREQ_HIGH_LEVEL
from agents.dynamic_auto_encoder import DynamicAutoEncoderAgent
from agents.rl_agent import DDPGAgent


IMAGE_RECEIVED = None
def fnc_img_callback(msg):
    global IMAGE_RECEIVED
    IMAGE_RECEIVED = msg

STATE_OBS_RECEIVED = None
def fnc_img_callback1(msg):
    global STATE_OBS_RECEIVED
    STATE_OBS_RECEIVED = msg


def reward1(np_state_obs_received):
    body_angle = np_state_obs_received[0]
    linear_velocity = np_state_obs_received[1]
    position = np_state_obs_received[2]
    dist2tgt_speed_accel = np_state_obs_received[3]

    '''
    linear_velocity[0]  +: backward (moving away from the target)   -: forward (getting close to the target)
    linear_velocity[1]  +: moving to left                           -: move to right
    linear_velocity[2]  +: downward vertically                      -: upward
    ''' 
    reward = linear_velocity[0]  

    return reward



if __name__ == '__main__':
    '''
    Input: image frame
    Outputs: previous state, action, reward, state_estimate
    '''
    rospy.init_node('high_level_decision_maker')
    sub_image = rospy.Subscriber('/airsim_node/camera_frame', Image, fnc_img_callback)
    sub_state_observation = rospy.Subscriber('/airsim_node/state_obs_values', Float32MultiArray, fnc_img_callback1)

    pub_transition = rospy.Publisher('/decision_maker_node/state_est_transition', Float32MultiArray, queue_size=1) # prev_state_est, action, reward, next_state_est
    pub_target = rospy.Publisher('/decision_maker_node/target', Twist, queue_size=1) # prev_state_est, action, reward, next_state_est
    rate=rospy.Rate(FREQ_HIGH_LEVEL)


    # msg init. the msg is to send out numpy array.
    msg_mat_transition = Float32MultiArray()
    msg_mat_transition.layout.dim.append(MultiArrayDimension())
    msg_mat_transition.layout.dim.append(MultiArrayDimension())
    msg_mat_transition.layout.dim[0].label = "height"
    msg_mat_transition.layout.dim[1].label = "width"
    

    state_estimator = DynamicAutoEncoderAgent(SETTING, train=False)
    rl_agent = DDPGAgent(SETTING) #, train=False)


    count = 0
    pre_state_est = np.zeros(N_STATE_DIM)
    prev_np_state_estimate = np.zeros(N_STATE_DIM)
    prev_np_action = np.zeros(N_ACT_DIM)
    taget_msg = Twist()

    while not rospy.is_shutdown():
        count += 1

        # Load the saved Model every 10 iteration
        if count%10 == 0:
            try:
                state_estimator.load_the_model(777, SETTING['name'])
                rl_agent.load_the_model(777, SETTING['name'])
            except:
                print('An error in loading the saved model. Two possible reasons: 1. no saved model, 2. both of nodes simultaneously try to access the file together')

        if IMAGE_RECEIVED is not None and STATE_OBS_RECEIVED is not None:

            with torch.no_grad(): 
                ### Update the state estimate ###
                np_im = np.frombuffer(IMAGE_RECEIVED.data, dtype=np.uint8).reshape(IMAGE_RECEIVED.height, IMAGE_RECEIVED.width, -1)
                np_im = np.array(np_im)
                np_im = cv2.resize(np_im, SETTING['encoder_image_size'], interpolation = cv2.INTER_AREA)
                np_state_estimate = state_estimator.step(np_im, prev_np_action).squeeze()

                ### Get action first ###
                prev_torch_state_estimate = torch.FloatTensor(prev_np_state_estimate).to(DEVICE)
                action = rl_agent.get_exploration_action(prev_torch_state_estimate).squeeze()

            taget_msg.linear.x = action[0]
            taget_msg.linear.y = action[1]
            taget_msg.linear.z = action[2]
            taget_msg.angular.x = action[3]

            pub_target.publish(taget_msg)

            

                

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
            
        rate.sleep()

