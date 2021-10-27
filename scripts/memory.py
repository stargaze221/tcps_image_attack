#!/usr/bin/env python3

from collections import deque
import random
import numpy as np
import itertools



class ImageTargetBuffer:

    def __init__(self, size):
        self.buffer = deque(maxlen=size)
        self.maxSize = size
        self.len = 0

    def add(self, obs_img, tgt_box):
        """
        adds a particular pair of image and target box in the memory buffer
        """
        pair = (obs_img, tgt_box)
        self.len += 1
        if self.len > self.maxSize:
            self.len = self.maxSize
        self.buffer.append(pair)

    def sample(self, count):
        """
        samples a random batch from the replay memory buffer
        :param count: batch size
        :return: batch (numpy array)
        """
        batch = []
        count = min(count, self.len)
        batch = random.sample(self.buffer, count)

        obs_img_arr = np.float32([arr[0] for arr in batch])
        tgt_box_arr = np.float32([arr[1] for arr in batch])
        
        return obs_img_arr, tgt_box_arr

    def len(self):
        return self.len



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


class TransitionBuffer:

    def __init__(self, size):
        self.buffer = deque(maxlen=size)
        self.maxSize = size
        self.len = 0

    def sample(self, count):
        """
        samples a random batch from the replay memory buffer
        :param count: batch size
        :return: batch (numpy array)
        """
        batch = []
        count = min(count, self.len)
        batch = random.sample(self.buffer, count)

        s_arr = np.float32([arr[0] for arr in batch])
        a_arr = np.float32([arr[1] for arr in batch])
        r_arr = np.float32([arr[2] for arr in batch])
        s1_arr = np.float32([arr[3] for arr in batch])
        done_arr = np.float32([arr[4] for arr in batch])

        return s_arr, a_arr, r_arr, s1_arr, done_arr

    def len(self):
        return self.len

    def add(self, s, a, r, s1, done):
        """
        adds a particular transaction in the memory buffer
        :param s: current state
        :param a: action taken
        :param r: reward received
        :param s1: next state
        :param done: if terminal
        :return:
        """
        transition = (s, a, r, s1, done)
        self.len += 1
        if self.len > self.maxSize:
            self.len = self.maxSize
        self.buffer.append(transition)


class SingleTrajectoryBuffer:
    
    def __init__(self, n_window_size):
        self.obs_img_memory = deque(maxlen=n_window_size)
        self.tgt_memory = deque(maxlen=n_window_size)
        self.state_est_memory = deque(maxlen=n_window_size)
        self.t = 0
        self.len = 0

    def add(self, obs_img, tgt_box, state_est):
        self.obs_img_memory.append(obs_img)
        self.tgt_memory.append(tgt_box)
        self.state_est_memory.append(state_est)
        self.len +=1
    
    def sample(self, n_windows):

        batch_obs_img_stream = []
        batch_tgt_stream = []
        batch_state_est_stream = []
        length = len(self.obs_img_memory)
        start = np.random.choice(length-n_windows)
        stop = min(start + n_windows, length)
        batch_obs_img_stream.append(list(itertools.islice(self.obs_img_memory, start, stop)))
        batch_tgt_stream.append(list(itertools.islice(self.tgt_memory, start, stop)))
        batch_state_est_stream.append(list(itertools.islice(self.state_est_memory, start, stop)))
            
        return np.array(batch_obs_img_stream), np.array(batch_tgt_stream), np.array(batch_state_est_stream)



if __name__ == "__main__":
    memory = ImageBuffer(1000)

    for i in range(100):
        frame = np.random.rand(448,448,3)
        print(i, frame.shape)
        memory.add(frame)

        minibatch = memory.sample(7)
        print('minibatch', len(minibatch), minibatch.shape)
