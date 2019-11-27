import tensorflow as tf
import numpy as np

class EpsScheduler(object):
    """Determines an action according to an epsilon greedy strategy with annealing epsilon"""
    def __init__(self, eps_initial=1, eps_final=0.1, eps_final_frame=0.01, 
                 eps_evaluation=0.0, eps_annealing_frames=1000000, 
                 replay_memory_start_size=50000, max_frames=25000000):
        """
        Args:
            dqn: DQN object
            n_actions: Integer, number of possible actions
            eps_initial: Float, Exploration probability for the first 
                replay_memory_start_size frames
            eps_final: Float, Exploration probability after 
                replay_memory_start_size + eps_annealing_frames frames
            eps_final_frame: Float, Exploration probability after max_frames frames
            eps_evaluation: Float, Exploration probability during evaluation
            eps_annealing_frames: Int, Number of frames over which the 
                exploration probabilty is annealed from eps_initial to eps_final
            replay_memory_start_size: Integer, Number of frames during 
                which the agent only explores
            max_frames: Integer, Total number of frames shown to the agent
        """
        self.eps_initial = eps_initial
        self.eps_final = eps_final
        self.eps_final_frame = eps_final_frame
        self.eps_evaluation = eps_evaluation
        self.eps_annealing_frames = eps_annealing_frames
        self.replay_memory_start_size = replay_memory_start_size
        self.max_frames = max_frames
        
        # Slopes and intercepts for exploration decrease
        self.slope = -(self.eps_initial - self.eps_final)/self.eps_annealing_frames
        self.intercept = self.eps_initial - self.slope*self.replay_memory_start_size
        self.slope_2 = -(self.eps_final - self.eps_final_frame)/(self.max_frames - self.eps_annealing_frames - self.replay_memory_start_size)
        self.intercept_2 = self.eps_final_frame - self.slope_2*self.max_frames

    def get_eps(self, frame_number, evaluation=False):
        """
        Args:
            frame_number: Integer, number of the current frame
            evaluation: A boolean saying whether the agent is being evaluated
        Returns:
           epsilon indicating exploration that should be done at this stage
        """
        eps = 0.0
        if evaluation:
            eps = self.eps_evaluation
        elif frame_number < self.replay_memory_start_size:
            eps = self.eps_initial
        elif frame_number >= self.replay_memory_start_size and frame_number < self.replay_memory_start_size + self.eps_annealing_frames:
            eps = self.slope*frame_number + self.intercept
        elif frame_number >= self.replay_memory_start_size + self.eps_annealing_frames:
            eps = self.slope_2*frame_number + self.intercept_2
        return eps        

class DQN:
    """Implements a Deep Q Network"""

    LAYER_IDS = ["conv1", "conv2", "conv3", "conv4", "denseAdvantage", 
                "denseAdvantageBias", "denseValue", "denseValueBias"]


    # pylint: disable=too-many-instance-attributes
    
    def __init__(self, n_actions:int, eps_scheduler:EpsScheduler, hidden=1024, learning_rate=0.00001, 
                 frame_height=84, frame_width=84, agent_history_length=4):
        """
        Args:
            n_actions: Integer, number of possible actions
            hidden: Integer, Number of filters in the final convolutional layer. 
                    This is different from the DeepMind implementation
            learning_rate: Float, Learning rate for the Adam optimizer
            frame_height: Integer, Height of a frame of an Atari game
            frame_width: Integer, Width of a frame of an Atari game
            agent_history_length: Integer, Number of frames stacked together to create a state
        """
        self.n_actions = n_actions
        self.eps_scheduler = eps_scheduler
        self.hidden = hidden
        self.learning_rate = learning_rate
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.agent_history_length = agent_history_length
        
        self.input = tf.placeholder(shape=[None, self.frame_height, 
                                           self.frame_width, self.agent_history_length], 
                                    dtype=tf.float32)
        # Normalizing the input
        self.inputscaled = self.input/255
        
        # Convolutional layers
        self.conv1 = tf.layers.conv2d(
            inputs=self.inputscaled, filters=32, kernel_size=[8, 8], strides=4,
            kernel_initializer=tf.variance_scaling_initializer(scale=2),
            padding="valid", activation=tf.nn.relu, use_bias=False, name='conv1')
        self.conv2 = tf.layers.conv2d(
            inputs=self.conv1, filters=64, kernel_size=[4, 4], strides=2, 
            kernel_initializer=tf.variance_scaling_initializer(scale=2),
            padding="valid", activation=tf.nn.relu, use_bias=False, name='conv2')
        self.conv3 = tf.layers.conv2d(
            inputs=self.conv2, filters=64, kernel_size=[3, 3], strides=1, 
            kernel_initializer=tf.variance_scaling_initializer(scale=2),
            padding="valid", activation=tf.nn.relu, use_bias=False, name='conv3')
        self.conv4 = tf.layers.conv2d(
            inputs=self.conv3, filters=hidden, kernel_size=[7, 7], strides=1, 
            kernel_initializer=tf.variance_scaling_initializer(scale=2),
            padding="valid", activation=tf.nn.relu, use_bias=False, name='conv4')
        
        # Splitting into value and advantage stream
        self.valuestream, self.advantagestream = tf.split(self.conv4, 2, 3)
        self.valuestream = tf.layers.flatten(self.valuestream)
        self.advantagestream = tf.layers.flatten(self.advantagestream)
        self.advantage = tf.layers.dense(
            inputs=self.advantagestream, units=self.n_actions,
            kernel_initializer=tf.variance_scaling_initializer(scale=2), name="advantage")
        self.value = tf.layers.dense(
            inputs=self.valuestream, units=1, 
            kernel_initializer=tf.variance_scaling_initializer(scale=2), name='value')
        
        # Combining value and advantage into Q-values as described above
        self.q_values = self.value + tf.subtract(self.advantage, tf.reduce_mean(self.advantage, axis=1, keepdims=True))
        self.best_action = tf.argmax(self.q_values, 1)
        
        # The next lines perform the parameter update. This will be explained in detail later.
        
        # targetQ according to Bellman equation: 
        # Q = r + gamma*max Q', calculated in the function learn()
        self.target_q = tf.placeholder(shape=[None], dtype=tf.float32)
        # Action that was performed
        self.action = tf.placeholder(shape=[None], dtype=tf.int32)
        # Q value of the action that was performed
        self.Q = tf.reduce_sum(tf.multiply(self.q_values, tf.one_hot(self.action, self.n_actions, dtype=tf.float32)), axis=1)
        
        # Parameter updates
        self.loss = tf.reduce_mean(tf.losses.huber_loss(labels=self.target_q, predictions=self.Q))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.update = self.optimizer.minimize(self.loss)

    def get_action(self, sess, frame_number, state, evaluation):
            # session: A tensorflow session object
            # state: A (84, 84, 4) sequence of frames of an Atari game in grayscale
        eps = self.eps_scheduler.get_eps(frame_number, evaluation)

        if np.random.rand(1) < eps:
            return np.random.randint(0, self.n_actions)
        return sess.run(self.best_action, feed_dict={self.input:[state]})[0]  


class Networks:
    HIDDEN = 1024                    # Number of filters in the final convolutional layer. The output 
                                    # has the shape (1,1,1024) which is split into two streams. Both 
                                    # the advantage stream and value stream have the shape 
                                    # (1,1,512). This is slightly different from the original 
                                    # implementation but tests I did with the environment Pong 
                                    # have shown that this way the score increases more quickly
    LEARNING_RATE = 0.00001          # Set to 0.00025 in Pong for quicker results. 
                                    # Hessel et al. 2017 used 0.0000625

    def __init__(self, action_n:int)->None:
        # main DQN and target DQN networks:
        with tf.variable_scope('mainDQN'):
            self.main_dqn = DQN(action_n, Networks.HIDDEN, Networks.LEARNING_RATE) 
        with tf.variable_scope('targetDQN'):
            self.target_dqn = DQN(action_n, Networks.HIDDEN)       

        self.saver = tf.train.Saver()    

        self.main_dqn_vars = tf.trainable_variables(scope='mainDQN')
        self.target_dqn_vars = tf.trainable_variables(scope='targetDQN')
        self._setup_tensorboard()

    def _setup_tensorboard(self):
        # ### Tensorboard
        # Setting up tensorboard summaries for the loss, the average reward, the evaluation score and the network parameters to observe the learning process:

        # Scalar summaries for tensorboard: loss, average reward and evaluation score
        with tf.name_scope('Performance'):
            self.loss_ph = tf.placeholder(tf.float32, shape=None, name='loss_summary')
            self.loss_summary = tf.summary.scalar('loss', self.loss_ph)
            self.reward_ph = tf.placeholder(tf.float32, shape=None, name='reward_summary')
            self.reward_summary = tf.summary.scalar('reward', self.reward_ph)
            self.eval_scope_ph = tf.placeholder(tf.float32, shape=None, name='evaluation_summary')
            self.eval_scope_summary = tf.summary.scalar('evaluation_score', self.eval_scope_ph)

        self.performance_summaries = tf.summary.merge([self.loss_summary, self.reward_summary])

        # Histogramm summaries for tensorboard: parameters
        with tf.name_scope('Parameters'):
            all_param_summaries = []
            for i, Id in enumerate(DQN.LAYER_IDS):
                with tf.name_scope('mainDQN/'):
                    main_dqn_kernal = tf.summary.histogram(Id, tf.reshape(self.main_dqn_vars[i], shape=[-1]))
                all_param_summaries.extend([main_dqn_kernal])
        self.param_summaries = tf.summary.merge(all_param_summaries)


    def _update_target_vars(self):
        update_ops = []
        for i, var in enumerate(self.main_dqn_vars):
            copy_op = self.target_dqn_vars[i].assign(var.value())
            update_ops.append(copy_op)
        return update_ops
            
    def update_target_network(self, sess):
        """
        Args:
            sess: A Tensorflow session object
        Assigns the values of the parameters of the main network to the 
        parameters of the target network
        """
        update_ops = self._update_target_vars()
        for copy_op in update_ops:
            sess.run(copy_op)      

