#!/usr/bin/env python3

from collections import deque
import random
import numpy as np


class ImageBuffer:

    def __init__(self, size):
            self.buffer = deque(maxlen=size)
            self.maxSize = size
            self.len = 0

    def add(self, obs_img):
        """
        adds a frame of image in the memory buffer
        """
        self.len += 1
        if self.len > self.maxSize:
            self.len = self.maxSize
        self.buffer.append(obs_img)

    def sample(self, count):
        """
        samples a random batch from the replay memory buffer
        :param count: batch size
        :return: batch (numpy array)
        """
        batch = []
        count = min(count, self.len)
        batch = random.sample(self.buffer, count)
        
        return batch

    def len(self):
        return self.len



if __name__ == "__main__":
    memory = ImageBuffer(1000)

    for i in range(100):
        frame = np.random.rand(448,448,3)
        print(i, frame.shape)
        memory.add(frame)

        minibatch = memory.sample(7)
        print('minibatch', len(minibatch), minibatch.shape)