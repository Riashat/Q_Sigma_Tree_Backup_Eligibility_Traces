"""
CORRECT
"""


import sys
sys.path.insert(0, "/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Project_652/Code/Linear_Approximator/Eligibility_Traces/Accumulating_Traces/")
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

from lib import plotting
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler
from collections import namedtuple

import matplotlib.pyplot as plt

import pyrl.basis.fourier as fourier
import pyrl.basis.rbf as rbf
import pyrl.basis.tilecode as tilecode




"""
Environment
"""


from collections import defaultdict
from lib.envs.windy_gridworld import WindyGridworldEnv
from lib import plotting
env = WindyGridworldEnv()

# env = gym.envs.make("MountainCar-v0")




"""
Feature Extactor
"""
observation_examples = np.array([env.observation_space.sample() for x in range(1)])
# scaler = sklearn.preprocessing.StandardScaler()
# scaler.fit(observation_examples)


#convert states to a feature representation:
#used an RBF sampler here for the feature map
featurizer = sklearn.pipeline.FeatureUnion([
        ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
        ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
        ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
        ("rbf4", RBFSampler(gamma=0.5, n_components=100))
        ])
# featurizer.fit(scaler.transform(observation_examples))

featurizer.fit(observation_examples)



# def featurize_state(state):
# 	#state = np.array(state)
# 	num_actions = env.action_space.n

# 	all_features = np.zeros(shape=(400))
# 	all_features = np.array([all_features]).T

# 	for a in range(num_actions):
# 		sprime, _, _, _ = env.step(a)
# 		features = featurizer.transform(sprime).T
# 		all_features = np.append(all_features, features, axis=1)

# 	all_features = all_features[:, 1:]

# 	return all_features


def featurize_state(state):
	# state = np.array([state])
	# scaled = scaler.transform([state])
	# featurized = featurizer.transform(scaled)
	featurized = featurizer.transform(state)
	return featurized[0]





def behaviour_policy_epsilon_greedy(theta, epsilon, nA):
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        phi = featurize_state(observation)
        q_values = np.dot(theta.T, phi)
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn


def create_greedy_policy(theta, epsilon, nA):
    def policy_fn(observation):
        A = np.zeros(nA, dtype=float) * epsilon / nA
        phi = featurize_state(observation)
        q_values = np.dot(theta.T, phi)
        best_action = np.argmax(q_values)
        A[best_action] = 1
        return A
    return policy_fn





from numpy.random import binomial
def binomial_sigma(p):
	sample = binomial(n=1, p=p)
	return sample


"""
Agent policies
"""

def epsilon_greedy_policy(theta, epsilon, nA):
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        phi = featurize_state(observation)
        q_values = np.dot(theta.T, phi)
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn





def true_online_tree_backup_lambda(env, theta, num_episodes, discount_factor=1.0, epsilon=0.1, epsilon_decay=1.0, alpha=0.1, lambda_param=1):

	
	stats = plotting.EpisodeStats(
		episode_lengths=np.zeros(num_episodes),
		episode_rewards=np.zeros(num_episodes))  

	# theta = np.zeros(shape=(400, env.action_space.n))

	num_experiments = 1000

	cumulative_errors = np.zeros(shape=(num_episodes, 1))

	for i_episode in range(num_episodes):

		print "Episode Number, True Online Tree Backup (lambda):", i_episode

		policy = epsilon_greedy_policy(theta, epsilon * epsilon_decay**i_episode, env.action_space.n)
		state = env.reset()
		action_probs = policy(state)
		action = np.random.choice(np.arange(len(action_probs)), p=action_probs)


		#initialising eligibility traces
		eligibility = np.zeros(shape=(theta.shape[0],env.action_space.n))

		next_action = None


		#previous_q_value
		theta_prev = theta
		features_state = featurize_state(state)
		previous_q_values = np.dot(theta_prev.T, features_state)
		previous_q_values_state_action = previous_q_values[action]



		for t in itertools.count():

			if next_action is None:
				action_probs = policy(state)
				action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
			else:
				action = next_action


			next_state, reward, done, _ = env.step(action)

			if done:
				eligibility = np.zeros(shape=(theta.shape[0]))
				print "Theta", theta
				break		

			stats.episode_rewards[i_episode] += reward
			stats.episode_lengths[i_episode] = t

			features_state = featurize_state(state)
			q_values = np.dot(theta.T, features_state)

			q_values_state_action = q_values[action]

			next_action_probs = policy(next_state)
			next_action = np.random.choice(np.arange(len(next_action_probs)), p = next_action_probs)

			next_features_state = featurize_state(next_state)
			next_q_values = np.dot(theta.T, next_features_state)
			next_q_values_state_action = next_q_values[next_action]


			V = np.sum(next_action_probs * next_q_values)
			Delta = reward + discount_factor * V - previous_q_values_state_action

			rms_error = np.sqrt(np.sum((Delta)**2)/num_experiments)
			cumulative_errors[i_episode, :] += rms_error


			next_action_probability = next_action_probs[next_action]
			
			eligibility[:, action] = eligibility[:, action] + alpha * features_state

			for s in range(features_state.shape[0]):
				for a in range(env.action_space.n):
					eligibility[s, a] = discount_factor * lambda_param * next_action_probability * eligibility[s, a]  - alpha * discount_factor * lambda_param * next_action_probability * eligibility[s,a] * np.dot(features_state.T, features_state)


			theta[:, action] = theta[:, action] + Delta * eligibility[:, action] + alpha * (previous_q_values_state_action - q_values_state_action) * features_state



			previous_q_values_state_action = next_q_values_state_action
			state = next_state

	return stats, cumulative_errors







def main():

	print "True Online Tree Backup (lambda)"

	
	# theta = np.random.normal(size=(400))
	theta = np.zeros(shape=(400, env.action_space.n))

	num_episodes = 1000
	smoothing_window = 1

	stats_sarsa_tb_lambda, cumulative_errors = true_online_tree_backup_lambda(env, theta, num_episodes)
	rewards_smoothed_stats_tb_lambda = pd.Series(stats_sarsa_tb_lambda.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()	
	cum_rwd = rewards_smoothed_stats_tb_lambda
	cum_err = cumulative_errors
	np.save('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Project_652/Code/Linear_Approximator/Eligibility_Traces/Accumulating_Traces/WindyGrid_Results/'  + 'True_Online_Tree_Backup_RBF_Cum_Rwd_2' + '.npy', cum_rwd)
	np.save('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Project_652/Code/Linear_Approximator/Eligibility_Traces/Accumulating_Traces/WindyGrid_Results/'  + 'True_Online_Tree_Backup_RBF_Cum_Err_2' + '.npy', cum_err)
	plotting.plot_episode_stats(stats_sarsa_tb_lambda)
	env.close()



	
if __name__ == '__main__':
	main()



