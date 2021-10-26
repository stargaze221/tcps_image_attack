#!/usr/bin/env python3

import torch
from torch.utils.tensorboard import SummaryWriter


SETTING = {}
SETTING.update({'yolov5_param_path':'yolov5s.pt'})
SETTING.update({'env_name': 'env1'})
SETTING.update({'name': 'NIPS_Env1_'+'Petros'})

SETTING.update({'alpha': 0.01})
SETTING.update({'lr_img_gen':0.001, 'lr_img_discrim':0.0012, 'lr_sys_id':0.0032, 'lr_actor':0.0016, 'lr_critic':0.0064, 'betas':(0.5, 0.9)})
SETTING.update({'n_batch_img':3, 'n_traj_img':1, 'n_batch_ddpg':8, 'n_window':30})

SETTING.update({'LAMBDA_COORD':0.001, 'LAMBDA_NOOBJ':0.001, 'AdvGen_Wt':1.0})

# Loss function parameters
SETTING.update({'max_trajectory_buffer':30, 'max_transition_buffer':100*10})
SETTING.update({'max_episodes':31, 'image_size':(448,448), 'action_dim':4, 'action_lim':1.0, 'state_dim':32})
SETTING.update({'encoder_image_size':(224,224)})


N_ACT_DIM = 4
N_STATE_DIM = 32
N_WINDOW = 16

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
N_MINIBATCH = 4

FREQ_HIGH_LEVEL = 5
FREQ_MID_LEVEL = 10
FREQ_LOW_LEVEL = 20


WRITER = SummaryWriter()
