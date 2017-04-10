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

# import pyrl.basis.fourier as fourier
# import pyrl.basis.rbf as rbf
# import pyrl.basis.tilecode as tilecode


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


def featurize_state(state):
	# state = np.array([state])
	# scaled = scaler.transform([state])
	# featurized = featurizer.transform(scaled)
	featurized = featurizer.transform(state)
	return featurized[0]



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




def behaviour_policy_Boltzmann(theta, tau, nA):
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * tau / nA
        phi = featurize_state(observation)
        q_values = np.dot(theta.T, phi)
        exp_tau = q_values / tau
        policy = np.exp(exp_tau) / np.sum(np.exp(exp_tau), axis=0)
        A = policy

        return A
    return policy_fn











def q_sigma_lambda_off_policy_static_sigma(env, theta, num_episodes, discount_factor=1.0, epsilon=0.1, epsilon_decay=1.0, alpha=0.5, lambda_param=0.1):

	
	stats = plotting.EpisodeStats(
		episode_lengths=np.zeros(num_episodes),
		episode_rewards=np.zeros(num_episodes))  


	num_experiments = 1000

	cumulative_errors = np.zeros(shape=(num_episodes, 1))

	tau = 1
	# tau_decay = 0.999

	for i_episode in range(num_episodes):

		print "Episode Number, Q (sigma, lambda) Off Policy, Static Sigma:", i_episode

		policy = epsilon_greedy_policy(theta, epsilon * epsilon_decay**i_episode, env.action_space.n)
		off_policy = behaviour_policy_Boltzmann(theta, tau * epsilon_decay**i_episode, env.action_space.n)

		state = env.reset()

		#initialising eligibility traces
		eligibility = np.zeros(shape=(theta.shape[0],env.action_space.n))

		next_action = None



		for t in itertools.count():

			if next_action is None:
				action_probs = policy(state)
				action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
			else:
				action = next_action

			next_state, reward, done, _ = env.step(action)
			stats.episode_rewards[i_episode] += reward
			stats.episode_lengths[i_episode] = t

			features_state = featurize_state(state)
			q_values = np.dot(theta.T, features_state)
			q_values_state_action = q_values[action]

			next_action_probs = off_policy(next_state)
			next_action = np.random.choice(np.arange(len(next_action_probs)), p = next_action_probs)
			next_action_probability = next_action_probs[next_action]


			next_features_state = featurize_state(next_state)
			next_q_values = np.dot(theta.T, next_features_state)
			next_q_values_state_action = next_q_values[next_action]


			V = np.sum (next_action_probs * next_q_values)


			probability = 0.5
			sigma = binomial_sigma(probability)
			Sigma_Effect = sigma * next_q_values_state_action + (1 - sigma) * V			

			td_target = reward + discount_factor * Sigma_Effect
			td_delta = td_target - q_values_state_action


			rms_error = np.sqrt(np.sum((td_delta)**2)/num_experiments)
			cumulative_errors[i_episode, :] += rms_error


			eligibility[:, action] = eligibility[:, action] + features_state


			for s in range(features_state.shape[0]):
				for a in range(env.action_space.n):
					#eligibility[s, a] = eligibility[s, a] * discount_factor * lambda_param * next_action_probability
					eligibility[s, a] = ( 1 - sigma ) * eligibility[s, a] * discount_factor * lambda_param * next_action_probability    +     sigma * discount_factor * lambda_param * eligibility[s, a]					


			theta[:, action] += alpha * eligibility[:, action] * td_delta


			if done:
				# print "Theta", theta
				break			

			state = next_state

	return stats, cumulative_errors









def q_sigma_lambda_off_policy_dynamic_sigma(env, theta, num_episodes, discount_factor=1.0, epsilon=0.1, epsilon_decay=1.0, alpha=0.5, lambda_param=0.1):

	
	stats = plotting.EpisodeStats(
		episode_lengths=np.zeros(num_episodes),
		episode_rewards=np.zeros(num_episodes))  

	num_experiments = 1000
	cumulative_errors = np.zeros(shape=(num_episodes, 1))

	sigma = 1
	sigma_decay = 0.995

	tau = 1


	for i_episode in range(num_episodes):

		print "Episode Number, Q (sigma, lambda) Off Policy, Dynamic Sigma:", i_episode

		policy = epsilon_greedy_policy(theta, epsilon * epsilon_decay**i_episode, env.action_space.n)
		off_policy = behaviour_policy_Boltzmann(theta, tau * epsilon_decay**i_episode, env.action_space.n)
		state = env.reset()

		#initialising eligibility traces
		eligibility = np.zeros(shape=(theta.shape[0],env.action_space.n))

		next_action = None

    	# tau = tau * tau_decay
    	# if tau < 0.0001:
    	# 	tau = 0.0001


		for t in itertools.count():

			if next_action is None:
				action_probs = policy(state)
				action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
			else:
				action = next_action

			next_state, reward, done, _ = env.step(action)
			stats.episode_rewards[i_episode] += reward
			stats.episode_lengths[i_episode] = t

			features_state = featurize_state(state)
			q_values = np.dot(theta.T, features_state)
			q_values_state_action = q_values[action]

			next_action_probs = off_policy(next_state)
			next_action = np.random.choice(np.arange(len(next_action_probs)), p = next_action_probs)
			next_action_probability = next_action_probs[next_action]


			next_features_state = featurize_state(next_state)
			next_q_values = np.dot(theta.T, next_features_state)
			next_q_values_state_action = next_q_values[next_action]


			on_policy_next_action_probs = policy(next_state)
			on_policy_next_action = np.random.choice(np.arange(len(on_policy_next_action_probs)), p = on_policy_next_action_probs)
			V = np.sum (on_policy_next_action_probs * next_q_values)


			Sigma_Effect = sigma * next_q_values_state_action + (1 - sigma) * V			

			td_target = reward + discount_factor * Sigma_Effect
			td_delta = td_target - q_values_state_action


			rms_error = np.sqrt(np.sum((td_delta)**2)/num_experiments)
			cumulative_errors[i_episode, :] += rms_error


			next_action_probability = next_action_probs[next_action]
			eligibility[:, action] = eligibility[:, action] + features_state


			for s in range(features_state.shape[0]):
				for a in range(env.action_space.n):
					#eligibility[s, a] = eligibility[s, a] * discount_factor * lambda_param * next_action_probability
					eligibility[s, a] = ( 1 - sigma ) * eligibility[s, a] * discount_factor * lambda_param * next_action_probability    +     sigma * discount_factor * lambda_param * eligibility[s, a]					


			theta[:, action] += alpha * eligibility[:, action] * td_delta


			if done:
				sigma = sigma * sigma_decay

				if sigma < 0.0001:
					sigma = 0.0001
				break

			state = next_state


	return stats, cumulative_errors







def main():

	print "Q(sigma, lambda) Off Policy"

	theta = np.zeros(shape=(400, env.action_space.n))
	
	num_episodes = 1000
	smoothing_window = 1

	# stats_sarsa_tb_lambda, cumulative_errors = q_sigma_lambda_off_policy_static_sigma(env, theta, num_episodes)
	# rewards_smoothed_stats_tb_lambda = pd.Series(stats_sarsa_tb_lambda.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()	
	# cum_rwd = rewards_smoothed_stats_tb_lambda
	# cum_err = cumulative_errors
	# np.save('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Project_652/Code/Linear_Approximator/Eligibility_Traces/Accumulating_Traces/WindyGrid_Results/'  + 'Q_sigma_lambda_OffPolicy_Static_Sigma_RBF_Cum_Rwd' + '.npy', cum_rwd)
	# np.save('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Project_652/Code/Linear_Approximator/Eligibility_Traces/Accumulating_Traces/WindyGrid_Results/'  + 'Q_sigma_lambda_OffPolicy_Static_Sigma_RBF_Cum_Err' + '.npy', cum_err)
	# plotting.plot_episode_stats(stats_sarsa_tb_lambda)
	# env.close()



	stats_sarsa_tb_lambda, cumulative_errors = q_sigma_lambda_off_policy_dynamic_sigma(env, theta, num_episodes)
	rewards_smoothed_stats_tb_lambda = pd.Series(stats_sarsa_tb_lambda.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()	
	cum_rwd = rewards_smoothed_stats_tb_lambda
	cum_err = cumulative_errors
	np.save('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Project_652/Code/Linear_Approximator/Eligibility_Traces/Accumulating_Traces/WindyGrid_Results/'  + 'Q_sigma_lambda_OffPolicy_Dynamic_Sigma_RBF_Cum_Rwd' + '.npy', cum_rwd)
	np.save('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Project_652/Code/Linear_Approximator/Eligibility_Traces/Accumulating_Traces/WindyGrid_Results/'  + 'Q_sigma_lambda_OffPolicy_Dynamic_Sigma_RBF_Cum_Err' + '.npy', cum_err)
	plotting.plot_episode_stats(stats_sarsa_tb_lambda)
	env.close()



	


if __name__ == '__main__':
	main()



