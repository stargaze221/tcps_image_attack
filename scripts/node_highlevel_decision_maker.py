#!/usr/bin/env python3
# generates the attack to send to the tello drone

import rospy
from std_msgs.msg import Float32MultiArray        # See https://gist.github.com/jarvisschultz/7a886ed2714fac9f5226
from std_msgs.msg import MultiArrayDimension      # See http://docs.ros.org/api/std_msgs/html/msg/MultiArrayLayout.html
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
import numpy as np
import torch
import cv2


from setting_params import N_ACT_DIM, N_STATE_DIM, DEVICE, SETTING, FREQ_HIGH_LEVEL
from agents.dynamic_auto_encoder import DynamicAutoEncoderAgent
from agents.rl_agent import DDPGAgent

from torch.utils.tensorboard import SummaryWriter

IMAGE_RECEIVED = None
def fnc_img_callback(msg):
    global IMAGE_RECEIVED
    IMAGE_RECEIVED = msg

STATE_OBS_RECEIVED = None
def fnc_img_callback1(msg):
    global STATE_OBS_RECEIVED
    STATE_OBS_RECEIVED = msg

RESET_START_RECEIVED = Bool()
RESET_START_RECEIVED.data = False
def fnc_callback(msg):
    global RESET_START_RECEIVED
    RESET_START_RECEIVED = msg

LOSS_MON_IMAGE_TRAIN_RECEIVED = None
def fnc_loss1_callback(msg):
    global LOSS_MON_IMAGE_TRAIN_RECEIVED
    LOSS_MON_IMAGE_TRAIN_RECEIVED = msg

LOSS_MON_HIGHLEVEL_TRAIN_RECEIVED = None
def fnc_loss2_callback(msg):
    global LOSS_MON_HIGHLEVEL_TRAIN_RECEIVED
    LOSS_MON_HIGHLEVEL_TRAIN_RECEIVED = msg


def reward1(np_state_obs_received):
    body_angle = np_state_obs_received[0]
    linear_velocity = np_state_obs_received[1]
    position = np_state_obs_received[2]
    dist2tgt_speed_accel = np_state_obs_received[3]

    dist2tgt = dist2tgt_speed_accel[0] 
    speed = dist2tgt_speed_accel[1] 


    other_finite_state = np_state_obs_received[4]
    done = other_finite_state[0]
    collision = other_finite_state[1]

    '''
    linear_velocity[0]  +: backward (moving away from the target)   -: forward (getting close to the target)
    linear_velocity[1]  +: moving to left                           -: move to right
    linear_velocity[2]  +: downward vertically                      -: upward
    ''' 
    if done > 0.5 and collision < 0.5:  # case 1: finised without collision
        reward = dist2tgt
        print('done and reward', reward)
    elif done > 0.5 and collision > 0.5: # case 2: finisehd wiht collision
        reward = dist2tgt
        print('collision and reward', reward)
    else:
        reward = dist2tgt # when it is not terminal uses speed. 
    return reward



if __name__ == '__main__':
    '''
    Input: image frame
    Outputs: previous state, action, reward, state_estimate
    '''
    rospy.init_node('high_level_decision_maker')
    sub_image = rospy.Subscriber('/airsim_node/camera_frame', Image, fnc_img_callback)
    sub_state_observation = rospy.Subscriber('/airsim_node/state_obs_values', Float32MultiArray, fnc_img_callback1)
    sub_reset_start = rospy.Subscriber('/airsim_node/reset_bool', Bool, fnc_callback)

    sub_loss_image_train = rospy.Subscriber('/image_attack_train_node/loss_monitor', Float32MultiArray, fnc_loss1_callback)
    sub_loss_highlevel_train = rospy.Subscriber('/decision_trainer_node/loss_monitor', Float32MultiArray, fnc_loss2_callback)

    pub_transition = rospy.Publisher('/decision_maker_node/state_est_transition', Float32MultiArray, queue_size=1) # prev_state_est, action, reward, next_state_est
    pub_target = rospy.Publisher('/decision_maker_node/target', Twist, queue_size=1) # prev_state_est, action, reward, next_state_est
    pub_reset_ack = rospy.Publisher('/decision_maker_node/reset_ack', Bool, queue_size=1)


    rate=rospy.Rate(FREQ_HIGH_LEVEL)


    # msg init. the msg is to send out numpy array.
    msg_mat_transition = Float32MultiArray()
    msg_mat_transition.layout.dim.append(MultiArrayDimension())
    msg_mat_transition.layout.dim.append(MultiArrayDimension())
    msg_mat_transition.layout.dim[0].label = "height"
    msg_mat_transition.layout.dim[1].label = "width"

    bool_ack_msg = Bool()
    bool_ack_msg.data = False
    

    state_estimator = DynamicAutoEncoderAgent(SETTING, train=False)
    rl_agent = DDPGAgent(SETTING) #, train=False)


    count = 0
    pre_state_est = np.zeros(N_STATE_DIM)
    prev_np_state_estimate = np.zeros(N_STATE_DIM)
    prev_np_action = np.zeros(N_ACT_DIM)
    taget_msg = Twist()

    # Log variables and writier
    iteration = 0

    log_count = 0
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
        
        count += 1

        #########################
        ### Reset Handshaking ###
        #########################
        ResetMsg = RESET_START_RECEIVED.data

        if ResetMsg: # at onset of receiving
            print('Reset Ack')
            bool_ack_msg.data = True
            pub_reset_ack.publish(bool_ack_msg)

        else:
            bool_ack_msg.data = False
            pub_reset_ack.publish(bool_ack_msg)


        # Load the saved Model every 10 iteration
        if count%FREQ_HIGH_LEVEL == 0:
            try:
                state_estimator.load_the_model()
                rl_agent.load_the_model()
            except:
                print('In high_level_decision_maker, model loading failed!')

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

            other_finite_state = np_state_obs_received[4]
            done = other_finite_state[0]
            collision = other_finite_state[1]



            ### State Transition to Pack ###
            # 1. previous state estimate   <-   "prev_np_state_estimate"
            # 2. action                    <-   "action"
            # 3. reward                    <-   "reward"
            # 4. current state estimate    <-   "np_state_estimate"

            np_transition = np.zeros((3, N_STATE_DIM))
            np_transition[0] = prev_np_state_estimate
            np_transition[1][:N_ACT_DIM] = action
            np_transition[1][-1] = reward
            np_transition[1][-2] = done
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

            ##################
            ### Log values ###
            ##################
            sum_reward += reward
            t_steps += 1

            if collision > 0.5:
                sum_n_collision +=1
                print('Collision!', reward)

            if done > 0.5:
                n_episode += 1
                print('Done!', reward)
                avg_reward = sum_reward/t_steps
                terminal_reward = reward
                writer.add_scalar('avg_reward', avg_reward, n_episode)
                writer.add_scalar('terminal_reward', terminal_reward, n_episode)
                writer.add_scalar('sum_n_collision', sum_n_collision, n_episode)
                writer.add_scalar('t_steps', t_steps, n_episode)
                t_steps = 0
                sum_reward = 0


        if LOSS_MON_IMAGE_TRAIN_RECEIVED is not None and LOSS_MON_HIGHLEVEL_TRAIN_RECEIVED is not None:

            iteration+= 1
            log_count+=1

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

        if log_count == 100:

            writer.add_scalar('loss_image_attack', sum_loss_image_attack/log_count, iteration)
            writer.add_scalar('loss_sys_id', sum_loss_sys_id/log_count, iteration)
            writer.add_scalar('loss_critic', sum_loss_critic/log_count, iteration)
            writer.add_scalar('loss_actor', sum_loss_actor/log_count, iteration)

            log_count = 0
            sum_loss_image_attack = 0
            sum_loss_sys_id = 0
            sum_loss_actor = 0
            sum_loss_critic = 0
            
        rate.sleep()

