import random
import argparse
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

from torchvision import utils

import gym
import gym.spaces


import numpy as np

log = gym.logger
log.set_level(gym.logger.INFO)

# Global Params
LATENT_VECTOR_SIZE = 100
DISCR_FILTERS = 64
GENER_FILTERS = 64
BATCH_SIZE = 16

# Dimension input image will be rescaled
IMAGE_SIZE = 64

LEARNING_RATE = 0.0001
REPORT_EVER_ITER = 100
SAVE_IMAGE_EVERY_ITER = 1000

# Custom Subclass to override gym.Observation wrapper
#
class InputWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(self, InputWrapper).__init__()
        self.env = env

    # Override the observation method
    def observation(self, observation):
        pass

