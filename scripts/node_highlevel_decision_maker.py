#!/usr/bin/env python3
# generates the attack to send to the tello drone
from agents.thompson_sampler import ThompsonSampler
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray        # See https://gist.github.com/jarvisschultz/7a886ed2714fac9f5226
from std_msgs.msg import MultiArrayDimension      # See http://docs.ros.org/api/std_msgs/html/msg/MultiArrayLayout.html
from sensor_msgs.msg import Image
import numpy as np
import cv2
import torch

from agents.dynamic_auto_encoder import DynamicAutoEncoderAgent
from agents.rl_agent import DDPGAgent
from agents.thompson_sampler import ThompsonSampler

from cv_bridge import CvBridge
from std_msgs.msg import Float32

from setting_params import N_ACT_DIM, N_STATE_DIM, DEVICE
from setting_params import SETTING

IMAGE_RECEIVED = None
def fnc_img_callback(msg):
    global IMAGE_RECEIVED
    IMAGE_RECEIVED = msg

ATTACK_LOSS_RECEIVED = None
def fnc_attack_loss_callback(msg):
    global ATTACK_LOSS_RECEIVED
    ATTACK_LOSS_RECEIVED = msg

if __name__ == '__main__':
    '''
    Input: 
    Outputs: previous state, action, reward, state_estimate
    '''

    rospy.init_node('high_level_decision_maker')

    sub_image = rospy.Subscriber('/tello_node/camera_frame', Image, fnc_img_callback)
    sub_attack_loss = rospy.Subscriber('/attack_generator_node/attack_loss', Float32, fnc_attack_loss_callback)

    pub_action_reward = rospy.Publisher('/decision_maker_node/action_reward', Float32MultiArray, queue_size=1)
    pub_prev_state_and_state_est = rospy.Publisher('/decision_maker_node/prev_state_and_state_est', Float32MultiArray, queue_size=1)
    pub_attack_lever = rospy.Publisher('/decision_maker_node/attack_lever', Float32, queue_size=1)
    
    rate=rospy.Rate(10)


    # msg init. the msg is to send out state value array.
    msg_action_reward = Float32MultiArray()
    msg_action_reward.layout.dim.append(MultiArrayDimension())
    msg_action_reward.layout.dim[0].label = "action_dim_+_1"
    msg_action_reward.layout.dim[0].size = N_ACT_DIM + 1
    msg_action_reward.layout.dim[0].stride = N_ACT_DIM + 1


    # msg init. the msg is to send out state value array.
    msg_prev_state_and_state_est = Float32MultiArray()
    msg_prev_state_and_state_est.layout.dim.append(MultiArrayDimension())
    msg_prev_state_and_state_est.layout.dim.append(MultiArrayDimension())
    msg_prev_state_and_state_est.layout.dim[0].label = "Prev_Cur"
    msg_prev_state_and_state_est.layout.dim[0].size = 2
    msg_prev_state_and_state_est.layout.dim[0].stride = 2*N_STATE_DIM
    msg_prev_state_and_state_est.layout.dim[1].label = "n_state"
    msg_prev_state_and_state_est.layout.dim[1].size = N_STATE_DIM
    msg_prev_state_and_state_est.layout.dim[1].stride = N_STATE_DIM
    msg_prev_state_and_state_est.layout.data_offset = 0

    # msg init.
    msg_attack_lever = Float32()


    pre_state_est = np.zeros(N_STATE_DIM)


    state_estimator = DynamicAutoEncoderAgent(SETTING, train=False)
    rl_agent = DDPGAgent(SETTING) #, train=False)
    thompson_sampler = ThompsonSampler(SETTING)

    mybridge = CvBridge()


    count = 0

    prev_np_state_estimate = np.zeros(N_STATE_DIM)
    prev_np_action = np.zeros(N_ACT_DIM)

    while not rospy.is_shutdown():
        count += 1

        # Load the saved Model every 10 iteration
        if count%10 == 0:
            try:
                state_estimator.load_the_model(777, SETTING['name'])
                rl_agent.load_the_model(777, SETTING['name'])
                thompson_sampler.load_the_model(777, SETTING['name'])
            except:
                print('An error in loading the saved model. Two possible reasons: 1. no saved model, 2. both of nodes simultaneously try to access the file together')

        if IMAGE_RECEIVED is not None:

            ### Get lever value ###
            msg_attack_lever.data = thompson_sampler.sample_lever_choice(prev_np_state_estimate, prev_np_action)

            ### Get action first ###
            prev_torch_state_estimate = torch.FloatTensor(prev_np_state_estimate).to(DEVICE)
            action = rl_agent.get_exploration_action(prev_torch_state_estimate).squeeze()
            
            ### Update the state estimate ###
            frame = mybridge.imgmsg_to_cv2(IMAGE_RECEIVED, desired_encoding='passthrough')
            frame = cv2.resize(frame, (224,224), cv2.IMREAD_UNCHANGED)  # (224, 224, 3)
            np_state_estimate = state_estimator.step(frame, action).squeeze()            

            #########################
            ### msg_action_reward ###
            #########################
            #action = np.random.rand(N_ACT_DIM)
            reward = np.random.rand(1)
            action_reward = np.concatenate((action, reward))
            msg_action_reward.data = action_reward.flatten().tolist()
            #print('msg_action_reward', msg_action_reward)

            ####################################
            ### msg_prev_state_and_state_est ###
            ####################################
            #state_est = np.random.rand(N_STATE_DIM)
            prev_state_and_state_est = np.concatenate((prev_np_state_estimate, np_state_estimate))
            msg_prev_state_and_state_est.data = prev_state_and_state_est.flatten().tolist()
            #print('msg_prev_state_and_state_est', msg_prev_state_and_state_est)

            

            # Publish
            pub_action_reward.publish(msg_action_reward)
            pub_prev_state_and_state_est.publish(msg_prev_state_and_state_est)
            pub_attack_lever.publish(msg_attack_lever)

            # Save the current state value.
            prev_np_state_estimate = np_state_estimate
            prev_np_action = action
            
        rate.sleep()