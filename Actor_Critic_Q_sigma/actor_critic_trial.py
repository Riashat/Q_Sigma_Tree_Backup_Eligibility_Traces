import gym
import itertools
import matplotlib
import numpy as np
import sys
import sklearn.pipeline
import sklearn.preprocessing

import pandas as pd
import sys
import random
import tensorflow as tf

from lib import plotting
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler
from collections import namedtuple


from collections import defaultdict
from lib.envs.cliff_walking import CliffWalkingEnv
from lib.envs.windy_gridworld import WindyGridworldEnv
from lib import plotting


#env = CliffWalkingEnv()
env=WindyGridworldEnv()

#samples from the state space to compute the features
observation_examples = np.array([env.observation_space.sample() for x in range(1)])

scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(observation_examples)


#convert states to a feature representation:
#used an RBF sampler here for the feature map
# featurizer = sklearn.pipeline.FeatureUnion([
#         ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
#         ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
#         ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
#         ("rbf4", RBFSampler(gamma=0.5, n_components=100))
#         ])

featurizer = sklearn.pipeline.FeatureUnion([
        ("rbf1", RBFSampler(gamma=0.5, n_components=100)),
        # ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
        # ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
        # ("rbf4", RBFSampler(gamma=0.5, n_components=100))
        ])

featurizer.fit(scaler.transform(observation_examples))



def featurize_state(state):

	state = np.array([state])

	scaled = scaler.transform([state])
	featurized = featurizer.transform(scaled)
	return featurized[0]

class PolicyEstimator():
    """
    Policy Function approximator. 
    """
    
    def __init__(self, learning_rate=0.01, scope="policy_estimator"):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.int32, [], "state")
            self.action = tf.placeholder(dtype=tf.int32, name="action")
            self.target = tf.placeholder(dtype=tf.float32, name="target")

            # This is just table lookup estimator
            state_one_hot = tf.one_hot(self.state, int(env.observation_space.n))
            self.output_layer = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(state_one_hot, 0),
                num_outputs=env.action_space.n,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer())

            self.action_probs = tf.squeeze(tf.nn.softmax(self.output_layer))
            self.picked_action_prob = tf.gather(self.action_probs, self.action)

            # Loss and train op
            self.loss = -tf.log(self.picked_action_prob) * self.target

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())
    
    def get_action_probs(self, state):
        sess = tf.get_default_session()
        return sess.run(self.action_probs, { self.state: state })

    def update(self, state, target, action, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = { self.state: state, self.target: target, self.action: action  }

        _, loss = sess.run([self.train_op, self.loss], feed_dict)

        return loss


class ValueEstimator():
	def __init__(self, env, state_features=None, learningrate=0.1, scope="value_estimator"):

		self.w_params = np.random.normal(size=(100, env.action_space.n))
		self.learningrate = learningrate


	def predict(self, state_features):
		value = np.dot(self.w_params.T, state_features)
		return value


	def update(self, td_error, state_features, action):
		self.w_params[:, action] = self.w_params[:, action] + self.learningrate * td_error * state_features
		updated_w_params = self.w_params

		return updated_w_params


class ValueEstimator():
    """
    Value Function approximator. 
    """
    
    def __init__(self, learning_rate=0.1, scope="value_estimator"):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.int32, [], "state")
            self.target = tf.placeholder(dtype=tf.float32, name="target")

            # This is just table lookup estimator
            state_one_hot = tf.one_hot(self.state, int(env.observation_space.n))
            self.output_layer = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(state_one_hot, 0),
                num_outputs=1,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer())

            self.value_estimate = tf.squeeze(self.output_layer)
            self.loss = tf.squared_difference(self.value_estimate, self.target)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())        
    
    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.value_estimate, { self.state: state })

    def update(self, state, target, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = { self.state: state, self.target: target }
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss



# def featurize_state(state):
# 	state_one_hot = tf.one_hot(state, int(env.observation_space.n))
# 	output_layer = tf.contrib.layers.fully_connected(
#                 inputs=tf.expand_dims(state_one_hot, 0),
#                 num_outputs=1,
#                 activation_fn=None,
#                 weights_initializer=tf.zeros_initializer())




def actor_critic(env, policy_estimator, value_estimator, num_episodes, learning_rate=0.1, discount_factor=1.0):
	    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))    

    for i_episode in range(num_episodes):
    	# Reset the environment and pick the fisrst action

    	print "Number of Episode", i_episode
    	# print "Episode Rewards", stats.episode_rewards[i_episode]

        state = env.reset()

		# One step in the environment
        for t in itertools.count():

        	#get action probabilities
        	action_probs = policy_estimator.get_action_probs(state)
        	# print "Action Probs", action_probs
        	action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        	# print "Action", action


        	next_state, reward, done, _ = env.step(action)
        	stats.episode_rewards[i_episode] += reward

        	features_state = featurize_state(state)
        	# print "Features State", features_state.shape


        	q_values_actions = value_estimator.predict(features_state)
        	# print "Q Function", q_values_actions


        	#predict function to return Q for all actions
        	q_value = q_values_actions[action]

        	features_state_next =featurize_state(next_state)
        	next_q_values_actions = value_estimator.predict(features_state_next)
        	next_q_value = next_q_values_actions[action]


        	td_target = reward + discount_factor * q_value
        	td_error = td_target - next_q_value       	


        	updated_w_params = value_estimator.update(td_error, features_state, action)
        	# print "Updated W Params", updated_w_params

        	policy_loss = policy_estimator.update(state, td_error, action)
        	# print "Policy Training Loss", policy_loss


        	if done:
        		break
        	state = next_state

    return stats




def main():

	tf.reset_default_graph()

	global_step = tf.Variable(0, name="global_step", trainable=False)
	policy_estimator = PolicyEstimator()
	value_estimator = ValueEstimator(env)

	num_episodes = 1000

	with tf.Session() as sess:
	    sess.run(tf.global_variables_initializer())
	    stats = actor_critic(env, policy_estimator, value_estimator, num_episodes)

	plotting.plot_episode_stats(stats, smoothing_window=25)

	env.close()

if __name__ == '__main__':
	main()



