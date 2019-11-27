import gym
from frame_processor import FrameProcessor
import numpy as np
import random

class Game(object):
    """Wrapper for the environment provided by gym"""
    def __init__(self, envName, is_atari:bool, no_op_steps=None, agent_history_length=4):
        self.env = gym.make(envName)
        self.process_frame = FrameProcessor()
        self.state = None
        self.last_lives = 0
        self.is_atari = is_atari
        self.no_op_steps = no_op_steps or (10 if is_atari else 0)
        self.agent_history_length = agent_history_length

    def reset(self, sess, evaluation=False):
        """
        Args:
            sess: A Tensorflow session object
            evaluation: A boolean saying whether the agent is evaluating or training
        Resets the environment and stacks four frames ontop of each other to 
        create the first state
        """
        frame = self.env.reset()
        self.last_lives = 0
        episode_done = True # Set to true so that the agent starts 
                                  # with a 'FIRE' action when evaluating
        if evaluation and self.no_op_steps >= 1:
            for _ in range(random.randint(1, self.no_op_steps)):
                frame, _, _, _ = self.env.step(1) # Action 'Fire'
        processed_frame = self.process_frame(sess, frame)
        self.state = np.repeat(processed_frame, self.agent_history_length, axis=2)
        
        return episode_done

    def step(self, sess, action):
        """
        Args:
            sess: A Tensorflow session object
            action: Integer, action the agent performs
        Performs an action and observes the reward and terminal state from the environment
        """
        new_frame, reward, terminal, info = self.env.step(action)
            
        if self.is_atari:
            if info['ale.lives'] < self.last_lives:
                episode_done = True
            else:
                episode_done = terminal
            self.last_lives = info['ale.lives']
        else:
            episode_done = terminal
        
        processed_new_frame = self.process_frame(sess, new_frame) 
        new_state = np.append(self.state[:, :, 1:], processed_new_frame, axis=2)  
        self.state = new_state
        
        return processed_new_frame, reward, terminal, episode_done, new_frame
