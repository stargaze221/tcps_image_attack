#!/usr/bin/env python3
import torch
import numpy as np
import torch.nn as nn

import os
from agents.util import get_target_index, make_grid
from setting_params import DEVICE, SETTING
from yolo_wrapper import YoloWrapper

from nn_networks.imageattack_nn import ImageAttackNetwork

GAMMA = 0.99
TAU = 0.001
EPS=1e-10
TARGET_CLASS_INDEX = 2
        

######################
### Image Attaker ####
######################
class ImageAttackTraniner:

    def __init__(self, setting_dict=SETTING):
            
        self.name = setting_dict['name']
        self.env_name = setting_dict['env_name']
        image_size = setting_dict['image_size']
        
        self.LAMBDA_COORD = setting_dict['LAMBDA_COORD']
        self.LAMBDA_NOOBJ = setting_dict['LAMBDA_NOOBJ']
        self.LAMBDA_L2 = setting_dict['LAMBDA_L2']
        self.LAMBDA_Var = setting_dict['LAMBDA_Var']
        self.attack_network = ImageAttackNetwork(image_size[0], image_size[1], 4).to(DEVICE)

        ### Yolo Model ###
        self.yolo_model = YoloWrapper(setting_dict['yolov5_param_path'])
        self.yolo_model.model.eval()
        self.anchors = self.yolo_model.model.yaml['anchors']
        self.stride = self.yolo_model.model.stride
        ### Attack Limit ###
        self.alpha = setting_dict['alpha']
        self.attack_network.train()
        self.optimizerG = torch.optim.Adam(self.attack_network.parameters(), setting_dict['lr_img_gen'], setting_dict['betas'])
        
    def save_the_model(self):
        if not os.path.exists('save/'+self.env_name+'/save/attack_network/'):
            os.makedirs('save/'+self.env_name+'/save/attack_network/')
        f_name = self.name + '_attack_network_param_' +  '_model.pth'
        #print('save/'+self.env_name+'/save/attack_network/'+f_name)
        torch.save(self.attack_network.state_dict(), 'save/'+self.env_name+'/save/attack_network/'+f_name)
        #print('ImageAttacker Model Saved')

    def update(self, obs_arr, tgt_arr, train=True):
        self.yolo_model.model.eval()
        self.attack_network.train()
        
        ### get batch sample ###
        X = torch.FloatTensor(obs_arr).to(DEVICE).permute(0, 3, 1, 2)  #.permute(0, 3, 1, 2).contiguous()
        X = X/255 # scale.
        Y = torch.FloatTensor(tgt_arr).to(DEVICE)
        X_attacked, X_adv = self.make_attacked_images(X,Y)
        loss = self.calculate_loss(X_attacked, X_adv, Y)
        ### Optimization Step ###
        self.optimizerG.zero_grad()
        loss.backward()
        self.optimizerG.step()

        loss_value = loss.item()

        del loss, X, Y
        torch.cuda.empty_cache()        

        return loss_value

    def make_attacked_images(self, X, Y):
        """
        X: minibatch image    [(1 x 3 x 448 x 448), ...]
        Y: target coordinates [(x, y, w, h), ...]
        """
        ### Draw target box ###
        Y = (Y + 1)/2  # [-1,1] -> [0,1] with some pertubation.
        Y = torch.clip(Y, 0, 1)
        Y[:,0] = torch.clip(Y[:,0]*448, 40, 400)  #x_ctr = int(tgt[0]*448)
        Y[:,1] = torch.clip(Y[:,1]*448, 40, 400)  #y_ctr = int(tgt[1]*448)
        Y[:,2] = Y[:,2]*200 + 50  # w = int(200*tgt[2] + 50)
        Y[:,3] = Y[:,3]*200 + 50  # h = int(200*tgt[3] + 50)
        X_adv = self.attack_network.get_attack_image(X, Y)
        X_attacked = torch.clip(X_adv*self.alpha + X, 0, 1)
        return X_attacked, X_adv

    def calculate_loss(self, x_attacked_image, x_adv, tgt):
        """
        X: minibatch image    [(1 x 3 x 448 x 448), ...]
        Y: target coordinates [(x, y, w, h), ...]
        """
        X = x_attacked_image
        Y = tgt
        n_minibatch, _, _, _ = X.size()
        
        ## Draw target box ###
        Y = (Y + 1)/2  # [-1,1] -> [0,1] with some pertubation.
        Y = torch.clip(Y, 0, 1)
        Y[:,0] = torch.clip(Y[:,0]*448, 40, 400)  #x_ctr = int(tgt[0]*448)
        Y[:,1] = torch.clip(Y[:,1]*448, 40, 400)  #y_ctr = int(tgt[1]*448)
        Y[:,2] = Y[:,2]*200 + 50  # w = int(200*tgt[2] + 50)
        Y[:,3] = Y[:,3]*200 + 50  # h = int(200*tgt[3] + 50)
        results = self.yolo_model.model(X)
        anchors = self.yolo_model.model.yaml['anchors']
        strides = self.yolo_model.model.stride #.cpu().numpy()

        ### Calcualte the loss ###
        nl = len(self.anchors)  # number of detection layers
        a = torch.tensor(self.anchors).float().view(nl, -1, 2)
        anchor_grid = a.clone().view(nl, 1, -1, 1, 1, 2).to(DEVICE)  # shape(nl,1,na,1,1,2)

        # error for loss function
        loss = 0
        error_xy = 0
        error_wh = 0 
        error_obj_confidence = 0
        error_no_obj_confidence = 0
        error_class = 0

        for s in range(n_minibatch):
            indice = get_target_index(self.anchors, self.stride, Y[s,:].unsqueeze(0)).long()
            i_tgt = indice[0][0].item() # layer index
            j_tgt = indice[0][1].item() # anchor index
            k_tgt = indice[0][2].item() # horizontal index
            l_tgt = indice[0][3].item() # vertical index
            Y_tgt = Y[s].unsqueeze(0)
            for i in range(nl):
                # Get the output tensor of the layer
                y = torch.sigmoid(results[1][i][s]).unsqueeze(0)
                if i == i_tgt:
                    ### Cllect loss terms ####
                    # 1. Coordinate loss 
                    _, _, ny, nx, _ = y.size()
                    grid = make_grid(nx, ny).to(DEVICE)
                    xy = (y[..., 0:2] * 2. - 0.5 + grid) * strides[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * anchor_grid[i]  # wh
                    ### Coordinates of the box
                    xy_box = xy[:,j_tgt,k_tgt,l_tgt]
                    wh_box = wh[:,j_tgt,k_tgt,l_tgt]
                    ### Coordinates of the target box
                    xy_box_tgt = Y_tgt[...,:2]
                    wh_box_tgt = Y_tgt[...,2:]
                    ### Coordinates errors 
                    error_xy += torch.sum((xy_box - xy_box_tgt)**2)
                    error_wh += torch.sum((wh_box**0.5 - wh_box_tgt**0.5)**2)
                    # 2. Object confidence loss 
                    Confidence = y[..., 4]  
                    Confidence_box = Confidence[:,j_tgt,k_tgt,l_tgt]
                    error_obj_confidence += torch.sum((Confidence_box-1)**2)
                    # 3. No Object confidence loss
                    no_obj_mask = torch.ones_like(Confidence)
                    no_obj_mask[0][j_tgt][k_tgt][l_tgt] = 0
                    error_no_obj_confidence += torch.sum(no_obj_mask*(Confidence**2))
                    # 4. Object class loss
                    class_three_boxes = y[:,:,k_tgt,l_tgt, 5:]
                    class_three_boxes_tgt = torch.zeros_like(class_three_boxes)
                    class_three_boxes_tgt[...,TARGET_CLASS_INDEX] = 1  #TARGET_CLASS_INDEX   It was 2
                    error_class += torch.sum((class_three_boxes-class_three_boxes_tgt)**2)
                else:
                    # 3. No Object confidence loss
                    Confidence = y[..., 4]  
                    error_no_obj_confidence += torch.sum(Confidence**2)
            loss += (self.LAMBDA_COORD*error_xy + self.LAMBDA_COORD*error_wh + error_obj_confidence + self.LAMBDA_NOOBJ*error_no_obj_confidence + error_class)/n_minibatch
            loss += torch.mean(x_adv**2)*self.LAMBDA_L2

            prob = (x_adv+1)/2
            entropy = -torch.mean(prob*torch.log(prob))

            loss += entropy*self.LAMBDA_Var*100



            # temp = -torch.mean(torch.log(torch.var(x_adv, 0)))*self.LAMBDA_Var
            # print('tmp', temp.item())
            print('entropy', entropy.item())

            return loss



class ImageAttacker():
    def __init__(self, setting_dict=SETTING):
        self.name = setting_dict['name']
        self.env_name = setting_dict['env_name']
        image_size = setting_dict['image_size']
        self.attack_network = ImageAttackNetwork(image_size[0], image_size[1], 4).to(DEVICE)
        ### Attack Limit ###
        self.alpha = setting_dict['alpha']
        self.attack_network.eval()
    
    def load_the_model(self):
        #print(os.getcwd())
        f_name = self.name + '_attack_network_param_' + '_model.pth'
        #print('save/'+self.env_name+'/save/attack_network/'+f_name)
        self.attack_network.load_state_dict(torch.load('save/'+self.env_name+'/save/attack_network/'+f_name))
        #print('ImageAttacker Model Loaded')

    def generate_attack(self, obs, tgt_box):
        self.attack_network.eval()
        obs = np.expand_dims(obs, 0)
        tgt_box = np.expand_dims(tgt_box, 0)  # <--- last index is for turning on and off.
        ### Generate Attacked Image ###
        image_torch = torch.FloatTensor(obs).to(DEVICE).permute(0, 3, 1, 2).contiguous().detach() #<--- To avoid MIXED MEMORY 
        image_torch = image_torch/255 # scale [0,255] to [0,1]
        action_torch = torch.FloatTensor(tgt_box).to(DEVICE).detach()
        """
        X: minibatch image    [(1 x 3 x 448 x 448), ...]
        Y: target coordinates [(x, y, w, h), ...]
        """
        Y = action_torch
        X = image_torch
        ## Draw target box ###
        Y = (Y + 1)/2  # [-1,1] -> [0,1] with some pertubation.
        Y = torch.clip(Y, 0, 1)
        Y[:,0] = torch.clip(Y[:,0]*448, 40, 400)  #x_ctr = int(tgt[0]*448)
        Y[:,1] = torch.clip(Y[:,1]*448, 40, 400)  #y_ctr = int(tgt[1]*448)
        Y[:,2] = Y[:,2]*200 + 50  # w = int(200*tgt[2] + 50)
        Y[:,3] = Y[:,3]*200 + 50  # h = int(200*tgt[3] + 50)
        X_adv = self.attack_network.get_attack_image(X, Y)
        X_attacked = torch.clip(X_adv*self.alpha + X, 0, 1)
        return X_attacked.detach().squeeze().permute(1, 2, 0).cpu().numpy()
   