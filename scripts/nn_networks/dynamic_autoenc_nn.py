import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DynamicAutoEncoderNetwork(nn.Module):

    def __init__(self, image_size, action_dim, state_dim):
        super(DynamicAutoEncoderNetwork, self).__init__()

        self.encoding_dim = state_dim
        self.height = image_size[0]
        self.width = image_size[1]
        self.action_dim = action_dim

        ngf = 8 # filter size for generator
        nc = 3 # n color chennal (RGB)

        ### Image Encoder ###
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 12, stride=5), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 8, stride=4), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 32, 4, stride=2), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 16, 3, stride=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Flatten(), #,
            nn.Linear(16, self.encoding_dim)  #<--- 784 is hard-coded as dependent on 448 x 448 x 3.    16 is hard-coded as dependent on 224 x 224 x 3.
        )

        ### State Predictor Given Prvious State and Current Encoded Image and Action ###
        self.gru_hidden_dim = self.encoding_dim
        self.rnn_layer = nn.GRU(input_size=self.encoding_dim + self.action_dim, hidden_size=self.gru_hidden_dim, batch_first=True) 

        ### Image Reconstructed from the State Predictors ###
        self.decoder = nn.Sequential(
            # input is Z, going into a convolutionc
            nn.ConvTranspose2d( self.gru_hidden_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 8, 5, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 5, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4), nn.ReLU(True),
            nn.ConvTranspose2d( ngf * 4, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4), nn.ReLU(True),
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2), nn.ReLU(True),
            nn.ConvTranspose2d( ngf * 2, ngf, 7, 3, 1, bias=False),
            nn.BatchNorm2d(ngf), nn.ReLU(True),
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )