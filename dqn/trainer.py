from dqn.dqn import Networks
from replay_memory import ReplayMemory
from eps_scheduler import EpsScheduler
from frame_processor import NoOpFrameProcessor, AtariFrameProcessor
from game import Game
from dqn import DQN, Networks

import tensorflow as tf


class Trainer:
    MAX_EPISODE_LENGTH = 18000       # Equivalent of 5 minutes of gameplay at 60 frames per second
    EPOCH_FRAME_COUNT = 200000          # Number of frames the agent sees between evaluations
    EVAL_STEPS = 10000               # Number of frames for one evaluation
    NETW_UPDATE_FREQ = 10000         # Number of chosen actions between updating the target network. 
                                    # According to Mnih et al. 2015 this is measured in the number of 
                                    # parameter updates (every four actions), however, in the 
                                    # DeepMind code, it is clearly measured in the number
                                    # of actions the agent choses
    DISCOUNT_FACTOR = 0.99           # gamma in the Bellman equation
    REPLAY_MEMORY_START_SIZE = 50000 # Number of completely random actions, 
                                    # before the agent starts learning
    MAX_FRAMES = 30000000            # Total number of frames the agent sees 
    MEMORY_SIZE = 1000000            # Number of transitions stored in the replay memory
    NO_OP_STEPS = 10                 # Number of 'NOOP' or 'FIRE' actions at the beginning of an 
                                    # evaluation episode
    UPDATE_FREQ = 4                  # Every four actions a gradient descend step is performed
    BATCH_SIZE = 32                          # Batch size

    def __init__(self, game:Game, networks:Networks)->None:
        self.game = game
        self.networks = networks
        self.global_frame_index = 0
        self.rewards = []
        self.loss_list = []

    def train():
        """Contains the training and evaluation loops"""
        my_replay_memory = ReplayMemory(size=Trainer.MEMORY_SIZE, batch_size=Trainer.BATCH_SIZE)   # (★)
        
        eps_sched = EpsScheduler(
            replay_memory_start_size=Trainer.REPLAY_MEMORY_START_SIZE, 
            max_frames=Trainer.MAX_FRAMES)
        
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            
            self.global_frame_index = 0
            self.rewards = []
            self.loss_list = []
            
            while self.global_frame_index < Trainer.MAX_FRAMES:
                self.epoch_frame_index = 0
                while self.epoch_frame_index < Trainer.EPOCH_FRAME_COUNT:
                    episode_done = self.game.reset(sess)
                    episode_reward_sum = 0
                    for _ in range(Trainer.MAX_EPISODE_LENGTH):
                        action = networks.main_dqn.get_action(sess, global_frame_index, game.state, evaluation=False)   
                        processed_new_frame, reward, terminal, episode_done, _ = game.step(sess, action)  
                        global_frame_index += 1
                        epoch_frame_index += 1
                        episode_reward_sum += reward
                        
                        # Clip the reward
                        clipped_reward = Trainer.clip_reward(reward)
                        
                        # (7★) Store transition in the replay memory
                        my_replay_memory.add_experience(action=action, 
                                                        frame=processed_new_frame[:, :, 0],
                                                        reward=clipped_reward, 
                                                        terminal=episode_done)   
                        
                        if global_frame_index % Trainer.UPDATE_FREQ == 0 and global_frame_index > Trainer.REPLAY_MEMORY_START_SIZE:
                            loss = Trainer.learn(sess, my_replay_memory, networks,
                                        Trainer.BATCH_SIZE, gamma = Trainer.DISCOUNT_FACTOR) # (8★)
                            loss_list.append(loss)
                        if global_frame_index % Trainer.NETW_UPDATE_FREQ == 0 and global_frame_index > Trainer.REPLAY_MEMORY_START_SIZE:
                            networks.update_target_network(sess) # (9★)
                        
                        if terminal:
                            terminal = False
                            break

                    rewards.append(episode_reward_sum)
                    
                    # Output the progress:
                    if len(rewards) % 10 == 0:
                        # Scalar summaries for tensorboard
                        if global_frame_index > REPLAY_MEMORY_START_SIZE:
                            summ = sess.run(performance_summaries, 
                                            feed_dict={loss_ph:np.mean(loss_list), 
                                                    reward_ph:np.mean(rewards[-100:])})
                            
                            SUMM_WRITER.add_summary(summ, global_frame_index)
                            loss_list = []
                        # Histogramm summaries for tensorboard
                        summ_param = sess.run(param_summaries)
                        SUMM_WRITER.add_summary(summ_param, global_frame_index)
                        
                        print(len(rewards), global_frame_index, np.mean(rewards[-100:]))
                        with open('rewards.dat', 'a') as reward_file:
                            print(len(rewards), global_frame_index, 
                                np.mean(rewards[-100:]), file=reward_file)
                
                ########################
                ###### Evaluation ######
                ########################
                terminal = True
                gif = True
                frames_for_gif = []
                eval_rewards = []
                evaluate_frame_number = 0
                
                for _ in range(EVAL_STEPS):
                    if terminal:
                        episode_done = game.reset(sess, evaluation=True)
                        episode_reward_sum = 0
                        terminal = False
                
                    # Fire (action 1), when a life was lost or the game just started, 
                    # so that the agent does not stand around doing nothing. When playing 
                    # with other environments, you might want to change this...
                    action = 1 if episode_done else \
                        networks.main_dqn.get_action(sess, global_frame_index, game.state, evaluation=True)
                    
                    processed_new_frame, reward, terminal, episode_done, new_frame = game.step(sess, action)
                    evaluate_frame_number += 1
                    episode_reward_sum += reward

                    if gif: 
                        frames_for_gif.append(new_frame)
                    if terminal:
                        eval_rewards.append(episode_reward_sum)
                        gif = False # Save only the first game of the evaluation as a gif
                        
                print("Evaluation score:\n", np.mean(eval_rewards))       
                try:
                    generate_gif(global_frame_index, frames_for_gif, eval_rewards[0], PATH)
                except IndexError:
                    print("No evaluation game finished")
                
                #Save the network parameters
                saver.save(sess, PATH+'/my_model', global_step=global_frame_index)
                frames_for_gif = []
                
                # Show the evaluation score in tensorboard
                summ = sess.run(eval_scope_summary, feed_dict={eval_scope_ph:np.mean(eval_rewards)})
                SUMM_WRITER.add_summary(summ, global_frame_index)
                with open('rewardsEval.dat', 'a') as eval_reward_file:
                    print(global_frame_index, np.mean(eval_rewards), file=eval_reward_file)


    def run_episod(self, networks:Networks, sess):
        for _ in range(Trainer.MAX_EPISODE_LENGTH):
            action = networks.main_dqn.get_action(sess, global_frame_index, game.state, evaluation=False)   
            processed_new_frame, reward, terminal, episode_done, _ = game.step(sess, action)  
            global_frame_index += 1
            epoch_frame_index += 1
            episode_reward_sum += reward
            
            # Clip the reward
            clipped_reward = Trainer.clip_reward(reward)
            
            # (7★) Store transition in the replay memory
            my_replay_memory.add_experience(action=action, 
                                            frame=processed_new_frame[:, :, 0],
                                            reward=clipped_reward, 
                                            terminal=episode_done)   
            
            if global_frame_index % Trainer.UPDATE_FREQ == 0 and global_frame_index > Trainer.REPLAY_MEMORY_START_SIZE:
                loss = Trainer.learn(sess, my_replay_memory, networks,
                            Trainer.BATCH_SIZE, gamma = Trainer.DISCOUNT_FACTOR) # (8★)
                loss_list.append(loss)
            if global_frame_index % Trainer.NETW_UPDATE_FREQ == 0 and global_frame_index > Trainer.REPLAY_MEMORY_START_SIZE:
                networks.update_target_network(sess) # (9★)
            
            if terminal:
                terminal = False
                break

    
    def clip_reward(reward):
        if reward > 0:
            return 1
        elif reward == 0:
            return 0
        else:
            return -1


    def learn(session, replay_memory, networks, batch_size, gamma):
        """
        Args:
            session: A tensorflow sesson object
            replay_memory: A ReplayMemory object
            main_dqn: A DQN object
            target_dqn: A DQN object
            batch_size: Integer, Batch size
            gamma: Float, discount factor for the Bellman equation
        Returns:
            loss: The loss of the minibatch, for tensorboard
        Draws a minibatch from the replay memory, calculates the 
        target Q-value that the prediction Q-value is regressed to. 
        Then a parameter update is performed on the main DQN.
        """

        main_dqn, target_dqn = networks.main_dqn, networks.target_dqn

        # Draw a minibatch from the replay memory
        states, actions, rewards, new_states, terminal_flags = replay_memory.get_minibatch()    
        # The main network estimates which action is best (in the next 
        # state s', new_states is passed!) 
        # for every transition in the minibatch
        arg_q_max = session.run(main_dqn.best_action, feed_dict={main_dqn.input:new_states})
        # The target network estimates the Q-values (in the next state s', new_states is passed!) 
        # for every transition in the minibatch
        q_vals = session.run(target_dqn.q_values, feed_dict={target_dqn.input:new_states})
        double_q = q_vals[range(batch_size), arg_q_max]
        # Bellman equation. Multiplication with (1-terminal_flags) makes sure that 
        # if the game is over, targetQ=rewards
        target_q = rewards + (gamma*double_q * (1-terminal_flags))
        # Gradient descend step to update the parameters of the main network
        loss, _ = session.run([main_dqn.loss, main_dqn.update], 
                            feed_dict={main_dqn.input:states, 
                                        main_dqn.target_q:target_q, 
                                        main_dqn.action:actions})
        return loss
