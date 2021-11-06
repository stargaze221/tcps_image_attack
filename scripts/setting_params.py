#!/usr/bin/env python3
import torch


### Here I will take rosparam that is name...

SETTING = {}
SETTING.update({'name': 'env1_testleftorright117'})
SETTING.update({'env_name': 'env1'})


# Yolo v5 parameter file path
SETTING.update({'yolov5_param_path':'yolov5s.pt'})

# Attack pertubation size
SETTING.update({'alpha': 0.10})

# Optimization learning rate
SETTING.update({'lr_img_gen':0.001, 'lr_img_discrim':0.0012, 'lr_sys_id':0.0032, 'lr_actor':0.0008, 'lr_critic':0.0016, 'betas':(0.5, 0.9)})

# Dimension of the data and the NN networks
#SETTING.update({'image_size':(448,448), 'encoder_image_size':(224,224)})
SETTING.update({'image_size':(448,448), 'encoder_image_size':(112,112)})
SETTING.update({'N_ACT_DIM': 4, 'ACTION_LIM':1.0, 'N_STATE_DIM': 16})

# Ornstein-Uhlenbeck
SETTING.update({'noise_theta':0.10, 'noise_sigma':0.2})

# Minibatch size and time window
SETTING.update({'N_MINIBATCH_IMG':3, 'N_MINIBATCH_DDPG':8, 'N_WINDOW':16})
SETTING.update({'N_SingleTrajectoryBuffer':1000, 'N_TransitionBuffer':1000, 'N_ImageBuffer':10000})

# Image attack loss function parameters
SETTING.update({'LAMBDA_COORD':0.001, 'LAMBDA_NOOBJ':0.001, 'LAMBDA_L2':1})

# Reward Choice
#SETTING.update({'reward_function': 'positive_distance'})
#SETTING.update({'reward_function': 'negative_distance'})
#SETTING.update({'reward_function': 'move-up'})
#SETTING.update({'reward_function': 'move-down'})
#SETTING.update({'reward_function': 'move-left'})
SETTING.update({'reward_function': 'move-right'})

# N_Episodes
SETTING.update({'N_Episodes': 200})



FREQ_HIGH_LEVEL = 5
FREQ_MID_LEVEL = 10
FREQ_LOW_LEVEL = 15
filter_coeff = 0.1

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")