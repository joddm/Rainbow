from collections import deque
import random
import atari_py
import torch
import cv2
from atari_wrappers import make_atari, wrap_deepmind


class Env():
  def __init__(self, args, episode_life=True):
    self.dtype = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor
    self.env = wrap_deepmind(make_atari('PongNoFrameskip-v4'), episode_life=episode_life, frame_stack=True, scale=True)
    self.ale = self.env.unwrapped.ale
    self.ale.setInt('random_seed', args.seed)
    self.ale.setInt('max_num_frames', args.max_episode_length)
    self.ale.loadROM(atari_py.get_game_path(args.game))  # ROM loading must be done after setting options
    self.actions = self.env.action_space.n
    self.training = True  # Consistent with model training mode

  def reset(self):
    state = torch.Tensor(self.env.reset()._force().transpose(2, 0, 1))
    return state

  def step(self, action):
    obs, reward, done, _ = self.env.step(action)
    state = torch.Tensor(obs._force().transpose(2, 0, 1))
    return state, reward, done

  def action_space(self):
    return self.actions

  def render(self):
    cv2.imshow('screen', self.ale.getScreenGrayscale())
    cv2.waitKey(1)

  def close(self):
    cv2.destroyAllWindows()
