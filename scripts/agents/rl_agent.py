from nn_networks.actor_critic_nn import Critic, Actor

from agents.util import OrnsteinUhlenbeckActionNoise, hard_update, soft_update
#from Utility.util import OrnsteinUhlenbeckActionNoise, hard_update, soft_update
import torch
#import numpy as np
#import torch.nn as nn
import torch.nn.functional as F
from setting_params import DEVICE

import os



GAMMA = 0.99
TAU = 0.001
EPS=1e-10

TARGET_CLASS_INDEX = 39


###########################
### High-level-attacker ###
###########################
class DDPGAgent:

    def __init__(self, setting_dict):

        self.name = setting_dict['name']
        self.env_name = setting_dict['env_name']

        """
        :param state_dim: Dimensions of state (int)
        :param action_dim: Dimension of action (int)
        :param action_lim: Used to limit action in [-action_lim,action_lim]
        :param ram: replay memory buffer object
        :return:
        """

        self.state_dim = setting_dict['N_STATE_DIM']
        self.action_dim = setting_dict['N_ACT_DIM']
        self.action_lim = setting_dict['ACTION_LIM']
        self.iter = 0
        self.noise = OrnsteinUhlenbeckActionNoise(self.action_dim, mu=0, theta=setting_dict['noise_theta'], sigma=setting_dict['noise_sigma'])

        self.actor = Actor(self.state_dim, self.action_dim, self.action_lim).to(DEVICE)
        self.target_actor = Actor(self.state_dim, self.action_dim, self.action_lim).to(DEVICE)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), setting_dict['lr_actor'], setting_dict['betas'])

        self.critic = Critic(self.state_dim, self.action_dim).to(DEVICE)
        self.target_critic = Critic(self.state_dim, self.action_dim).to(DEVICE)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), setting_dict['lr_critic'], setting_dict['betas'])

        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

    def get_exploitation_action(self, state):
        """
        gets the action from target actor added with exploration noise
        :param state: state (Numpy array)
        :return: sampled action (Numpy array)
        """
        action = self.target_actor.forward(state).detach()
        
        return action.data.cpu().numpy()

    def get_exploration_action(self, state):
        """
        gets the action from actor added with exploration noise
        :param state: state (Numpy array)
        :return: sampled action (Numpy array)
        """
        action = self.actor.forward(state).detach()
        #action = (action + 1)/2
        new_action = action.data.cpu().numpy() + (self.noise.sample() * self.action_lim)

        #self.noise.theta = self.noise.theta*0.99

        return new_action

    def update(self, s1,a1,r1,s2,done):

        s1 = torch.from_numpy(s1).to(DEVICE).float()
        a1 = torch.from_numpy(a1).to(DEVICE).float()
        r1 = torch.from_numpy(r1).to(DEVICE).float()
        s2 = torch.from_numpy(s2).to(DEVICE).float()
        done = torch.from_numpy(done).to(DEVICE).float()

        
        # ---------------------- optimize critic ----------------------
        # Use target actor exploitation policy here for loss evaluation
        a2 = self.target_actor.forward(s2).detach()
        next_val = torch.squeeze(self.target_critic.forward(s2, a2).detach())
        # y_exp = r + gamma*Q'( s2, pi'(s2))
        #y_expected = r1 + GAMMA*next_val
        #print('r1', r1.size())
        #print('next_val', next_val.size())


        y_expected = r1 + GAMMA*(next_val - done*next_val)  # y_expected = r1 + GAMMA*next_val
        # y_pred = Q( s1, a1)
        y_predicted = torch.squeeze(self.critic.forward(s1, a1))
        #print('y_predicted', y_predicted.size())
        #print('y_expected', y_expected.size())
        

        # compute critic loss, and update the critic
        loss_critic = F.smooth_l1_loss(y_predicted, y_expected)
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()

        # ---------------------- optimize actor ----------------------
        pred_a1 = self.actor.forward(s1)
        loss_actor = -1*torch.sum(self.critic.forward(s1, pred_a1))
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()

        soft_update(self.target_actor, self.actor, TAU)
        soft_update(self.target_critic, self.critic, TAU)

        return loss_actor.item(), loss_critic.item()
        

    def save_the_model(self):
        if not os.path.exists('save/'+self.env_name+'/save/ddpg/'):
            os.makedirs('save/'+self.env_name+'/save/ddpg/')
        f_name = self.name + '_actor_network_param_' + '_model.pth'
        torch.save(self.actor.state_dict(), 'save/'+self.env_name+'/save/ddpg/'+f_name)
        f_name = self.name + '_critic_network_param_' + '_model.pth'
        torch.save(self.critic.state_dict(), 'save/'+self.env_name+'/save/ddpg/'+f_name)
        #print('DDPGAgent Model Saved')

    def load_the_model(self):
        f_name = self.name + '_actor_network_param_'  + '_model.pth'
        self.actor.load_state_dict(torch.load('save/'+self.env_name+'/save/ddpg/'+f_name))
        f_name = self.name + '_critic_network_param_' + '_model.pth'
        self.critic.load_state_dict(torch.load('save/'+self.env_name+'/save/ddpg/'+f_name))
        #print('DDPGAgent Model Loaded')