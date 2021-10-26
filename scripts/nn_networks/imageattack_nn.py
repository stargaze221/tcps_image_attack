#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ImageAttackNetwork(nn.Module):

    def __init__(self, h, w, action_dim):
        super(ImageAttackNetwork, self).__init__()

        ngf = 16
        nc = 3

        self.encoding_dim = 64
        self.height = h
        self.width = w
        self.action_dim = action_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 12, stride=5), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 8, stride=4), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 32, 4, stride=2), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 16, 3, stride=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(784, self.encoding_dim)  #<--- 784 is hard-coded as dependent on 448 x 448 x 3.
        )

        # self.decoder = nn.Sequential(
        #     # input is Z, going into a convolutionc
        #     nn.ConvTranspose2d( self.encoding_dim, ngf * 8, 4, 1, 0, bias=False),
        #     nn.BatchNorm2d(ngf * 8), nn.ReLU(True),
        #     nn.ConvTranspose2d(ngf * 8, ngf * 8, 5, 2, 1, bias=False),
        #     nn.BatchNorm2d(ngf * 8), nn.ReLU(True),
        #     nn.ConvTranspose2d(ngf * 8, ngf * 4, 5, 2, 1, bias=False),
        #     nn.BatchNorm2d(ngf * 4), nn.ReLU(True),
        #     nn.ConvTranspose2d( ngf * 4, ngf * 4, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(ngf * 4), nn.ReLU(True),
        #     nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(ngf * 2), nn.ReLU(True),
        #     nn.ConvTranspose2d( ngf * 2, ngf, 7, 3, 1, bias=False),
        #     nn.BatchNorm2d(ngf), nn.ReLU(True),
        #     nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
        #     nn.Tanh()
        # )

        self.attack_generator = nn.Sequential(
            # input is Z, going into a convolutionc
            nn.ConvTranspose2d( self.encoding_dim + self.action_dim, ngf * 8, 4, 1, 0, bias=False),
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

    # def reconstruct_image(self, x_img):
    #     encoding = self.encoder(x_img)
    #     x = torch.unsqueeze(torch.unsqueeze(encoding, -1), -1)
    #     x = self.decoder(x)
    #     x = x[:, :, :self.width, :self.height] # Crop Image
    #     return x

    def get_attack_image(self, x_img, tgt):
        encoding = self.encoder(x_img)
        x = torch.cat([encoding, tgt], 1)
        x = torch.unsqueeze(torch.unsqueeze(x, -1), -1)
        x = self.attack_generator(x)
        x = x[:, :, :self.width, :self.height] # Crop Image
        return x





if __name__ == "__main__":
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = ImageAttackNetwork(h=448, w=448, action_dim=3).to(DEVICE)

    x = torch.FloatTensor(np.random.rand(1, 3, 448, 448)).to(DEVICE)
    print('x', x.size())

    act = torch.FloatTensor(np.random.rand(1,3)).to(DEVICE)
    print('act', act.size())

    img = generator.get_attack_image(x, act)
    print('img', img.size())