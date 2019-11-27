"""
Implementation of DeepMind's Deep Q-Learning by Fabio M. Graetz, 2018
If you have questions or suggestions, write me a mail fabiograetzatgooglemaildotcom
"""
import os
import random
import gym
import tensorflow as tf
import numpy as np
import imageio
from skimage.transform import resize

from game import Game
from dqn import DQN

TRAIN = False
IS_ATARI = True # is environment ALE/Atari in which case specific code is enabled
ENV_NAME = 'BreakoutDeterministic-v4'
#ENV_NAME = 'PongDeterministic-v4'  
# You can increase the learning rate to 0.00025 in Pong for quicker results

def generate_gif(frame_number, frames_for_gif, reward, path):
    """
        Args:
            frame_number: Integer, determining the number of the current frame
            frames_for_gif: A sequence of (210, 160, 3) frames of an Atari game in RGB
            reward: Integer, Total reward of the episode that es ouputted as a gif
            path: String, path where gif is saved
    """
    for idx, frame_idx in enumerate(frames_for_gif): 
        frames_for_gif[idx] = resize(frame_idx, (420, 320, 3), 
                                     preserve_range=True, order=0).astype(np.uint8)
        
    imageio.mimsave(f'{path}{"ATARI_frame_{0}_reward_{1}.gif".format(frame_number, reward)}', 
                    frames_for_gif, duration=1/30)

# Control parameters

PATH = "output/"                 # Gifs and checkpoints will be saved here
SUMMARIES = "summaries"          # logdir for tensorboard
RUNID = 'run_1'
os.makedirs(PATH, exist_ok=True)
os.makedirs(os.path.join(SUMMARIES, RUNID), exist_ok=True)
SUMM_WRITER = tf.summary.FileWriter(os.path.join(SUMMARIES, RUNID))


tf.reset_default_graph()

game = Game(envName=ENV_NAME, is_atari=IS_ATARI, no_op_steps=NO_OP_STEPS)

print("The environment has the following {} actions: {}".format(game.env.action_space.n, 
                                                                game.env.unwrapped.get_action_meanings()))




if TRAIN:
    train()


save_files_dict = {
    'BreakoutDeterministic-v4':("trained/breakout/", "my_model-15845555.meta"),
    'PongDeterministic-v4':("trained/pong/", "my_model-3217770.meta")
}


if not TRAIN:
    
    gif_path = "GIF/"
    os.makedirs(gif_path, exist_ok=True)

    trained_path, save_file = save_files_dict[ENV_NAME]

    eps_sched = EpsScheduler(
        replay_memory_start_size=REPLAY_MEMORY_START_SIZE, 
        max_frames=MAX_FRAMES)
    
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(trained_path+save_file)
        saver.restore(sess,tf.train.latest_checkpoint(trained_path))
        frames_for_gif = []
        episode_done = game.reset(sess, evaluation = True)
        episode_reward_sum = 0
        while True:
            game.env.render()
            action = 1 if episode_done else networks.main_dqn.get_action(sess, 0, game.state,  
                                                                                   evaluation = True)
            
            processed_new_frame, reward, terminal, episode_done, new_frame = game.step(sess, action)
            episode_reward_sum += reward
            frames_for_gif.append(new_frame)
            if terminal == True:
                break
        
        game.env.close()
        print("The total reward is {}".format(episode_reward_sum))
        print("Creating gif...")
        generate_gif(0, frames_for_gif, episode_reward_sum, gif_path)
        print("Gif created, check the folder {}".format(gif_path))

