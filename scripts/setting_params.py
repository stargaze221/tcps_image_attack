#!/usr/bin/env python3

import torch


# import names
# import json
# from pathlib import Path


# name =names.get_last_name()
# path_to_file = name+'.json'
# path = Path(path_to_file)

# if path.is_file():
#     with open(path_to_file, 'r') as fp:
#         SETTING = json.load(fp)
# else:
#     SETTING = {}
#     SETTING.update({'yolov5_param_path':'yolov5s.pt'})
#     SETTING.update({'env_name': 'env1'})
#     SETTING.update({'name': name})

#     SETTING.update({'alpha': 0.05})
#     SETTING.update({'lr_img_gen':0.001, 'lr_img_discrim':0.0012, 'lr_sys_id':0.0032, 'lr_actor':0.0016, 'lr_critic':0.0064, 'betas':(0.5, 0.9)})
#     SETTING.update({'n_batch_img':3, 'n_traj_img':1, 'n_batch_ddpg':8, 'n_window':30})

#     SETTING.update({'LAMBDA_COORD':0.001, 'LAMBDA_NOOBJ':0.002, 'AdvGen_Wt':1.0})

#     # Loss function parameters
#     SETTING.update({'max_trajectory_buffer':30, 'max_transition_buffer':100*10})
#     SETTING.update({'max_episodes':31, 'image_size':(448,448), 'action_dim':4, 'action_lim':1.0, 'state_dim':32})
#     SETTING.update({'encoder_image_size':(224,224)})

#     with open(path_to_file, 'w') as fp:
#         json.dump(SETTING, fp)

SETTING = {}
SETTING.update({'yolov5_param_path':'yolov5s.pt'})
SETTING.update({'env_name': 'env1'})
SETTING.update({'name': 'Yoon'})

SETTING.update({'alpha': 0.05})
SETTING.update({'lr_img_gen':0.001, 'lr_img_discrim':0.0012, 'lr_sys_id':0.0032, 'lr_actor':0.0016, 'lr_critic':0.0064, 'betas':(0.5, 0.9)})
SETTING.update({'n_batch_img':3, 'n_traj_img':1, 'n_batch_ddpg':8, 'n_window':30})

SETTING.update({'LAMBDA_COORD':0.001, 'LAMBDA_NOOBJ':0.001, 'AdvGen_Wt':1.0})

# Loss function parameters
SETTING.update({'max_trajectory_buffer':30, 'max_transition_buffer':100*10})
SETTING.update({'max_episodes':31, 'image_size':(448,448), 'action_dim':4, 'action_lim':1.0, 'state_dim':32})
SETTING.update({'encoder_image_size':(224,224)})


N_ACT_DIM = 4
N_STATE_DIM = 32
N_WINDOW = 8

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
N_MINIBATCH = 4

FREQ_HIGH_LEVEL = 5
FREQ_MID_LEVEL = 10
FREQ_LOW_LEVEL = 20
