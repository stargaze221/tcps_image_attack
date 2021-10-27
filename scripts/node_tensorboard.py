#!/usr/bin/env python3
import numpy as np
import rospy, cv2
import torch
import os

from std_msgs.msg import Float32MultiArray        # See https://gist.github.com/jarvisschultz/7a886ed2714fac9f5226
from std_msgs.msg import Float32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist

from agents.image_attack_agent import ImageAttacker
from setting_params import N_ACT_DIM, DEVICE, SETTING, FREQ_HIGH_LEVEL


import torch
from torch.utils.tensorboard import SummaryWriter



LOSS_MON_IMAGE_TRAIN_RECEIVED = None
def fnc_callback(msg):
    global LOSS_MON_IMAGE_TRAIN_RECEIVED
    LOSS_MON_IMAGE_TRAIN_RECEIVED = msg

LOSS_MON_HIGHLEVEL_TRAIN_RECEIVED = None
def fnc_callback1(msg):
    global LOSS_MON_HIGHLEVEL_TRAIN_RECEIVED
    LOSS_MON_HIGHLEVEL_TRAIN_RECEIVED = msg

STATE_OBSERVATION_RECEIVED = None
def fnc_callback2(msg):
    global STATE_OBSERVATION_RECEIVED
    STATE_OBSERVATION_RECEIVED = msg

STATE_EST_TRANSITION = None
def fnc_callback3(msg):
    global STATE_EST_TRANSITION
    STATE_EST_TRANSITION = msg

N_LOG_INTERVAL = 10

if __name__ == '__main__':
    rospy.init_node('tensorboard_node')
    
    sub_loss_image_train = rospy.Subscriber('/image_attack_train_node/loss_monitor', Float32MultiArray, fnc_callback)
    sub_loss_highlevel_train = rospy.Subscriber('/decision_trainer_node/loss_monitor', Float32MultiArray, fnc_callback1)
    sub_state_observation = rospy.Subscriber('/airsim_node/state_obs_values', Float32MultiArray, fnc_callback2)
    sub_state_est_transition = rospy.Subscriber('/decision_maker_node/state_est_transition', Float32MultiArray, fnc_callback3) 
    
    rate=rospy.Rate(FREQ_HIGH_LEVEL)

    iteration = 0

    count = 0
    sum_loss_image_attack = 0
    sum_loss_sys_id = 0
    sum_loss_actor = 0
    sum_loss_critic = 0

    writer = SummaryWriter()

    t_steps = 0
    sum_reward = 0
    sum_n_collision = 0
    n_episode = 0

    

    while not rospy.is_shutdown():

        if STATE_EST_TRANSITION is not None and STATE_OBSERVATION_RECEIVED is not None:

            t_steps+=1

            # Unpack the msg to get reward #
            height = STATE_EST_TRANSITION.layout.dim[0].size
            width = STATE_EST_TRANSITION.layout.dim[1].size
            np_transition = np.array(STATE_EST_TRANSITION.data).reshape((height, width))

            prev_np_state_estimate = np_transition[0]
            action = np_transition[1][:N_ACT_DIM]
            reward = np_transition[1][-1]
            done = np_transition[1][-2]
            np_state_estimate = np_transition[2]

            sum_reward += reward #<-- Accumulate the rewards for performance.

            # Unpack the msg to get other informations: done, collision, etc.
            height = STATE_OBSERVATION_RECEIVED.layout.dim[0].size
            width = STATE_OBSERVATION_RECEIVED.layout.dim[1].size
            np_state_obs_received = np.array(STATE_OBSERVATION_RECEIVED.data).reshape((height, width))
            
            body_angle = np_state_obs_received[0]
            linear_velocity = np_state_obs_received[1]
            position = np_state_obs_received[2]
            dist2tgt_speed_accel = np_state_obs_received[3]
            other_finite_state = np_state_obs_received[4]
            
            done = other_finite_state[0]
            collision = other_finite_state[1]

            if collision > 0.5:
                print('collision')
                sum_n_collision += 1
                print(reward)

            if done > 0.5:
                n_episode+=1
                writer.add_scalar('average reward', sum_reward/t_steps, iteration)
                writer.add_scalar('terminal reward', reward, iteration)
                writer.add_scalar('total collision counts', sum_n_collision, iteration)
                writer.add_scalar('t_steps', t_steps, iteration)
                writer.add_scalar('n_episode', n_episode, iteration)
                t_steps = 0
                sum_reward = 0
                


        if LOSS_MON_IMAGE_TRAIN_RECEIVED is not None and LOSS_MON_HIGHLEVEL_TRAIN_RECEIVED is not None:

            iteration += 1
            count+=1

            height = LOSS_MON_IMAGE_TRAIN_RECEIVED.layout.dim[0].size
            width = LOSS_MON_IMAGE_TRAIN_RECEIVED.layout.dim[1].size
            np_loss_image_train = np.array(LOSS_MON_IMAGE_TRAIN_RECEIVED.data).reshape((height, width))
            loss_image_attack = np_loss_image_train[0][0]
            sum_loss_image_attack+=loss_image_attack

            height = LOSS_MON_HIGHLEVEL_TRAIN_RECEIVED.layout.dim[0].size
            width = LOSS_MON_HIGHLEVEL_TRAIN_RECEIVED.layout.dim[1].size
            np_loss_highlevel_train = np.array(LOSS_MON_HIGHLEVEL_TRAIN_RECEIVED.data).reshape((height, width))
            loss_sys_id, loss_actor, loss_critic = (np_loss_highlevel_train[0][0], np_loss_highlevel_train[0][1], np_loss_highlevel_train[0][2])
            sum_loss_sys_id += loss_sys_id
            sum_loss_actor += loss_actor
            sum_loss_critic += loss_critic

        if count == 10:

            writer.add_scalar('loss_image_attack', sum_loss_image_attack/count, iteration)
            writer.add_scalar('loss_sys_id', sum_loss_sys_id/count, iteration)
            writer.add_scalar('loss_critic', sum_loss_critic/count, iteration)
            writer.add_scalar('loss_actor', sum_loss_actor/count, iteration)

            count = 0
            sum_loss_image_attack = 0
            sum_loss_sys_id = 0
            sum_loss_actor = 0
            sum_loss_critic = 0
        
        rate.sleep()