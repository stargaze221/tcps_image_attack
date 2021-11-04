from nn_networks.dynamic_autoenc_nn import DynamicAutoEncoderNetwork, SmallerDynamicAutoEncoderNetwork

import torch
import torch.nn.functional as F

import os
from setting_params import DEVICE



GAMMA = 0.99
TAU = 0.001
EPS=1e-10

TARGET_CLASS_INDEX = 39



################################
### DynamicAutoEncoderAgent ####
################################
class DynamicAutoEncoderAgent:
    '''
    Dynamic Auto Encoder
    '''
    def __init__(self, setting_dict, train=True):
        self.name = setting_dict['name']
        self.env_name = setting_dict['env_name']
        lr_estimator = setting_dict['lr_sys_id']
        betas = setting_dict['betas']
        image_size = setting_dict['encoder_image_size']
        action_dim = setting_dict['N_ACT_DIM']
        state_dim = setting_dict['N_STATE_DIM']

        if image_size == (112, 112):
            self.nn_model = SmallerDynamicAutoEncoderNetwork(image_size, action_dim, state_dim).to(DEVICE)
        else:
            self.nn_model = DynamicAutoEncoderNetwork(image_size, action_dim, state_dim).to(DEVICE)

        self.if_train = train

        if self.if_train:
            self.nn_model.train()
            self.optimizer = torch.optim.Adam(self.nn_model.parameters(), lr_estimator, betas)
        else:
            self.nn_model.eval()
        
        self.state = torch.rand(1, 1, state_dim).to(DEVICE) #<--- rnn layer h_0 of shape (num_layers * num_directions, batch, hidden_size)

    def predict_batch_images(self, stream_arr, state_est_arr, tgt_arr):

        n_window, n_width, n_height, n_channel = stream_arr.shape
        n_window, n_state = state_est_arr.shape
        n_window, n_tgt = tgt_arr.shape

        stream_arr = torch.FloatTensor(stream_arr).to(DEVICE).permute(0,3,1,2).contiguous()
        state_est_arr = torch.FloatTensor(state_est_arr).to(DEVICE)
        tgt_arr = torch.FloatTensor(tgt_arr).to(DEVICE)

        ### Encoding ###
        encoding_streams = self.nn_model.encoder(stream_arr)
        ### State Predictor ###
        #print('encoding_streams', encoding_streams.size())
        #print('tgt_arr', tgt_arr.size())
        x_stream = torch.cat([encoding_streams, tgt_arr], 1).unsqueeze(0)
        #print('x_stream', x_stream.size())
        h0 = state_est_arr[0].unsqueeze(0).unsqueeze(0)
        output, h_n = self.nn_model.rnn_layer(x_stream, h0)
        ### Decoding ###
        output = output.squeeze().unsqueeze(-1).unsqueeze(-1)
        pred_image_stream = self.nn_model.decoder(output)
        pred_image_stream = pred_image_stream[:, :, :self.nn_model.width, :self.nn_model.height] # Crop Image
        
        return pred_image_stream
        

    def get_rnn_states(self, image_stream, action_stream):
        ### Encoding ###
        encoding_streams = self.nn_model.encoder(image_stream)

        ### State Predictor ###
        x_stream = torch.cat([encoding_streams, action_stream], 1).unsqueeze(0)
        h0 = torch.zeros(1, 1, self.nn_model.gru_hidden_dim).to(DEVICE)
        rnn_states, h_n = self.nn_model.rnn_layer(x_stream, h0)

        return rnn_states.detach()

    def step(self, observation, action):
        self.nn_model.eval()

        ### Encoding ###
        image = torch.FloatTensor(observation).permute(2, 0, 1).unsqueeze(0).to(DEVICE).detach()
        image = image/255 # scale it down from [0, 255] to [0, 1]
        action = torch.FloatTensor(action).unsqueeze(0).to(DEVICE).detach()
        encoding = self.nn_model.encoder(image).detach()

        ### State Predictor ###
        x = torch.cat([encoding, action], 1).unsqueeze(0)
        new_state, hidden = self.nn_model.rnn_layer(x, self.state)
        self.state = new_state.detach()

        del image, action, encoding, hidden, new_state
        torch.cuda.empty_cache()

        return self.state.detach().cpu().numpy()
        

    def update(self, stream_arr, state_est_arr, tgt_arr):
        '''
        The system is learned from N_BATCH trajectories sampled from TRAJ_MEMORY and each of them are cropped with the same time WINDOW
        '''
        self.nn_model.train()

        stream_arr = stream_arr.squeeze()
        state_est_arr = state_est_arr.squeeze()
        tgt_arr = tgt_arr.squeeze()

        ### Predict One Step Future ###        
        pred_image_stream = self.predict_batch_images(stream_arr, state_est_arr, tgt_arr)
        stream_arr = torch.FloatTensor(stream_arr).to(DEVICE).permute(0,3,1,2).contiguous()
        
        ### Normalize Signals into [0, 1] ###
        tgt_prob_img = stream_arr/255 # <--- scale it into (0, 1) from (0, 255)
        rec_prob_img = (pred_image_stream + 1)/2

        #### Translate one step the target for calculating loss in prediction
        tgt_prob_img = tgt_prob_img[1:, :, :, :]
        rec_prob_img = rec_prob_img[:-1, :, :, :]
        
        #### Cross Entropy Loss ###
        loss = torch.mean(-(tgt_prob_img*torch.log(rec_prob_img + EPS)+(1-tgt_prob_img)*torch.log(1-rec_prob_img + EPS)))

        ### Update Model ###
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_val = loss.item()

        del rec_prob_img, tgt_prob_img, pred_image_stream, loss
        torch.cuda.empty_cache()

        return loss_val


    def save_the_model(self):
        if not os.path.exists('save/'+self.env_name+'/save/dynautoenc/'):
            os.makedirs('save/'+self.env_name+'/save/dynautoenc/')
        f_name = self.name + '_dynautoenc_network_param_' + '_model.pth'
        torch.save(self.nn_model.state_dict(), 'save/'+self.env_name+'/save/dynautoenc/'+f_name)
        #print('DynamicAutoEncoderAgent Model Saved')

    def load_the_model(self):
        f_name = self.name + '_dynautoenc_network_param_' +  '_model.pth'
        self.nn_model.load_state_dict(torch.load('save/'+self.env_name+'/save/dynautoenc/'+f_name))
        #print('DynamicAutoEncoderAgent Model Loaded')

