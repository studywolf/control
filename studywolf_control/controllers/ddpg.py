import tensorflow as tf
import numpy as np


#####################  hyper parameters  ####################

MAX_EPISODES = 1000
MAX_EP_STEPS = 500
LR_A = 0.001  # learning rate for actor
LR_C = 0.002  # learning rate for critic
GAMMA = 0.99  # reward discount
TAU = 0.001  # soft replacement
MEMORY_CAPACITY = 1000000
OUTPUT_GRAPH = True
RENDER = False
# ENV_NAME = 'FetchReach-v0'
ENV_NAME = 'Pendulum-v0'

LAYER1_SIZE = 400
LAYER2_SIZE = 300
A_LEARNING_RATE = 1e-4
C_LEARNING_RATE = 1e-3
BATCH_SIZE = 64


###############################  DDPG  ####################################

class ActorNetwork(object):
    def __init__(self, sess, state_dim, action_dim, action_bound):
        self.sess = sess
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.state_dim = state_dim

        with tf.variable_scope('Actor'):
            # actor network, input s, output a
            self.scaled_a, self.state_in = self._build_net( scope='eval_net', trainable=True)

            # actor target network, input s_, output a, get a_ for critic
            self.t_scaled_a, self.t_state_in = self._build_net(scope='target_net', trainable=False)

        # A collection is nothing but a named set of values.
        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval_net')
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target_net')

        # tf.assign(ref, value) : Update 'ref' by assigning 'value' to it.
        self.update_network_a = [tf.assign(t, (1 - TAU) * t + TAU * e)
                                 for t, e in zip(self.t_params, self.e_params)]

        # This gradient will be provided by the critic network
        self.a_gradient = tf.placeholder(tf.float32, [None, self.action_dim])

        with tf.variable_scope('policy_grads'):
            # ys = policy;
            # xs = policy's parameters;
            # a_grads = the gradients of the policy to get more Q
            # tf.gradients will calculate dys/dxs with a initial gradients for ys, so this is dq/da * da/dparams
            self.policy_grads = tf.gradients(ys=self.scaled_a, xs=self.e_params, grad_ys=-self.a_gradient)

        with tf.variable_scope('A_train'):
            opt = tf.train.AdamOptimizer(A_LEARNING_RATE)
            # (- learning rate) for ascent policy
            self.train_op = opt.apply_gradients(zip(self.policy_grads, self.e_params))

        self.num_trainable_vars = len(self.e_params) + len(self.t_params)

    def _build_net(self, scope, trainable):
        state_input = tf.placeholder(tf.float32, [None, self.state_dim])
        with tf.variable_scope(scope):
            # init_w = tf.random_normal_initializer(0., 0.3)
            init_w = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
            init_b = tf.constant_initializer(0.1)
            net_h1 = tf.layers.dense(state_input, LAYER1_SIZE, activation=tf.nn.relu,
                                     kernel_initializer=init_w, bias_initializer=init_b, name='l1',
                                     trainable=trainable)
            net_h2 = tf.layers.dense(net_h1, LAYER2_SIZE, activation=tf.nn.relu,
                                     kernel_initializer=init_w, bias_initializer=init_b, name='l2',
                                     trainable=trainable)
            with tf.variable_scope('a'):
                actions = tf.layers.dense(net_h2, self.action_dim, activation=tf.nn.tanh, kernel_initializer=init_w,
                                          bias_initializer=init_b, name='a', trainable=trainable)
                # Scale output to -action_bound to action_bound
                scaled_a = tf.multiply(actions, self.action_bound, name='scaled_a')
                # print('actions', actions, scaled_a)
        return scaled_a, state_input

    def learn(self, s, a_gradient):  # batch update
        self.sess.run(self.train_op, feed_dict={self.state_in: s, self.a_gradient: a_gradient})

    def update_target_network(self):
        self.sess.run(self.update_network_a)

    def predict(self, s):
        return self.sess.run(self.scaled_a, feed_dict={self.state_in: s})

    def predict_target(self, s_):
        return self.sess.run(self.t_scaled_a, feed_dict={self.t_state_in: s_})

class CriticNetwork(object):
    def __init__(self, sess, state_dim, action_dim):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        #self.gamma = GAMMA

        with tf.variable_scope('Critic'):
            # Input (s, a), output q
            self.q, self.state_input, self.action_input = self._build_net('eval_net', trainable=True)

            # Input (s_, a_), output q_ for q_target
            self.q_, self.t_state_input, self.t_action_input = self._build_net('target_net', trainable=False)  # target_q is based on a_ from Actor's target_net

        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval_net')
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target_net')

        self.update_network_c = [tf.assign(t, (1 - TAU) * t + TAU * e)
                                 for t, e in zip(self.t_params, self.e_params)]

        self.target_q = tf.placeholder(tf.float32, [None, 1])
        #with tf.variable_scope('target_q'):
            #self.target_q = self.R + self.gamma * self.q_

        with tf.variable_scope('TD_error'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.target_q, self.q))

        with tf.variable_scope('C_train'):
            self.train_op = tf.train.AdamOptimizer(C_LEARNING_RATE).minimize(self.loss)

        with tf.variable_scope('a_grad'):
            self.a_grads = tf.gradients(self.q, self.action_input)[0]  # tensor of gradients of each sample (None, a_dim)

    def _build_net(self, scope, trainable):
        state_input = tf.placeholder(tf.float32, [None, self.s_dim])
        action_input = tf.placeholder(tf.float32, [None, self.a_dim])
        with tf.variable_scope(scope):
            # init_w = tf.random_normal_initializer(0., 0.1)
            init_w = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
            init_b = tf.constant_initializer(0.1)

            net_h1 = tf.layers.dense(state_input, LAYER1_SIZE, activation=tf.nn.relu,
                                     kernel_initializer=init_w, bias_initializer=init_b, name='l1',
                                     trainable=trainable)

            with tf.variable_scope('l2'):
                w2_s = tf.get_variable('w2_s', [LAYER1_SIZE, LAYER2_SIZE], initializer=init_w, trainable=trainable)
                w2_a = tf.get_variable('w2_a', [self.a_dim, LAYER2_SIZE], initializer=init_w, trainable=trainable)
                b2 = tf.get_variable('b2', [1, LAYER2_SIZE], initializer=init_b, trainable=trainable)
                net_h2 = tf.nn.relu(tf.matmul(net_h1, w2_s) + tf.matmul(action_input, w2_a) + b2)

            with tf.variable_scope('q'):
                q = tf.layers.dense(net_h2, 1, kernel_initializer=init_w, bias_initializer=init_b,
                                    trainable=trainable)  # Q(s,a)
        return q, state_input, action_input

    def learn(self, s, a, tq):
        return self.sess.run([self.q, self.train_op], feed_dict={self.state_input: s, self.action_input: a, self.target_q: tq})

    def update_target_network(self):
        self.sess.run(self.update_network_c)

    def action_grad(self, s, a):
        return self.sess.run(self.a_grads, feed_dict={self.state_input: s, self.action_input: a})

    def predict(self, s, a):
        return self.sess.run(self.q, feed_dict={self.state_input: s, self.action_input: a})

    def predict_target(self, s_, a_):
        return self.sess.run(self.q_, feed_dict={self.t_state_input: s_, self.t_action_input: a_})


class ReplayBuffer(object):
    # init replay buffer - memory
    def __init__(self, buffer_size, buffer_dims):
        self.buffer_size = buffer_size
        self.memory = np.zeros((buffer_size, buffer_dims))
        self.pointer = 0

    # replace the old memory with new memory
    # store transition in memory set
    def store_transition(self, s, a, r, s_, done):
        transition = np.hstack((s, a, r, s_, done))
        index = self.pointer % self.buffer_size
        self.memory[index, :] = transition
        self.pointer += 1

    # sample a random minibatch of minibatch_size transitions from memory
    def minibatch_sample(self, minibatch_size):
        assert self.pointer >= minibatch_size, 'Memory (Experience replay buffer) is not used!'
        if self.pointer > self.buffer_size:
            random_index = np.random.choice(len(self.memory), size=minibatch_size)
        else:
            random_index = np.random.choice(self.pointer, size=minibatch_size)
        # print('rando',random_index, self.memory)
        # random_index = int(random_index)
        return self.memory[random_index, :]

class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


def exit_handler():
    print('process killed and plot')