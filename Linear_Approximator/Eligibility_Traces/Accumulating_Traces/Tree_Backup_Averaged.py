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
from lib.envs.cliff_walking import CliffWalkingEnv
from lib import plotting
env = CliffWalkingEnv()

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




"""
Online Tree Backup with Eligiblity Traces with Linear Function Approximator (RBF Kernel)
"""
def tree_backup_lambda(env, theta, num_episodes, discount_factor=1.0, alpha=0.1, epsilon=0.1, epsilon_decay=1.0, lambda_param=0.1):


	stats = plotting.EpisodeStats(
		episode_lengths=np.zeros(num_episodes),
		episode_rewards=np.zeros(num_episodes))  


	lambda_param = np.array([0.1, 0.15, 0.2, 0.4, 0.6, 0.8, 0.9, 1])
	alpha = np.array([0.1, 0.2, 0.4, 0.5])


	All_Rwd_Lambda = np.zeros(shape=(num_episodes, len(lambda_param)))
	All_Lambda_Alpha = np.zeros(shape=(len(lambda_param), len(alpha)))

	All_Error_Lambda = np.zeros(shape=(num_episodes, len(lambda_param)))	
	All_Error_Lambda_Alpha = np.zeros(shape=(len(lambda_param), len(alpha)))

	num_experiments = num_episodes

	

	for l in range(len(lambda_param)):

		print "Lambda Param", lambda_param[l]

		for alpha_param in range(len(alpha)):

			print "Alpha Param", alpha[alpha_param]

			theta = np.zeros(shape=(400, env.action_space.n))
			cumulative_errors = np.zeros(shape=(num_episodes))

			for i_episode in range(num_episodes):

				print "Number of Episodes, Tree Backup (lambda)", i_episode

				policy = epsilon_greedy_policy(theta, epsilon, env.action_space.n)
				state = env.reset()

				action_probs = policy(state)
				action = np.random.choice(np.arange(len(action_probs)), p=action_probs)	

				eligibility = np.zeros(shape=(theta.shape[0],env.action_space.n))
				# eligibility = defaultdict(lambda : np.zeros(env.action_space.n))	
				#initialising eligibility traces

				next_action = None

				for t in itertools.count():


					next_state, reward, done, _ = env.step(action)
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



					V = np.sum (next_action_probs * next_q_values)

					td_target = reward + discount_factor * V

					Delta = td_target - q_values_state_action

					rms_error = np.sqrt(np.sum((Delta)**2)/num_experiments)

					cumulative_errors[i_episode] += rms_error

					next_action_probability = next_action_probs[next_action]

					eligibility[:, action] = eligibility[:, action] + features_state



					for st in range(features_state.shape[0]):
						for at in range(env.action_space.n):
							eligibility[st, at] = eligibility[st, at] * discount_factor * lambda_param[l] * next_action_probability
					
					theta[:, action] += alpha[alpha_param] * eligibility[:, action] * Delta

					

					if done:
						print "Theta", theta
						break

					action = next_action
					state = next_state

			cum_rwd_per_episode = np.array([pd.Series(stats.episode_rewards).rolling(1, min_periods=1).mean()])
			cum_error_per_episode = cumulative_errors

			All_Rwd_Lambda[:, l] = cum_rwd_per_episode
			All_Error_Lambda[:, l] = cum_error_per_episode
			All_Lambda_Alpha[l, alpha_param] = cum_error_per_episode.T[-1]
			All_Error_Lambda_Alpha[l, alpha_param] = cum_error_per_episode.T[-1]


	return All_Rwd_Lambda, All_Lambda_Alpha, All_Error_Lambda, All_Error_Lambda_Alpha



def main():

	print "Tree Backup(lambda)"
	env = CliffWalkingEnv()
	Total_num_experiments = 2
	num_episodes = 30

	theta = np.zeros(shape=(400, env.action_space.n))

	lambda_param = np.array([0.1, 0.15, 0.2, 0.4, 0.6, 0.8, 0.9, 1])
	alpha = np.array([0.1, 0.2, 0.4, 0.5])

	Averaged_All_Rwd_Lambda = np.zeros(shape=(num_episodes, len(lambda_param)))
	Averaged_All_Lambda_Alpha = np.zeros(shape=(len(lambda_param), len(alpha)))
	Averaged_All_Error_Lambda = np.zeros(shape=(num_episodes, len(lambda_param)))	
	Averaged_All_Error_Lambda_Alpha = np.zeros(shape=(len(lambda_param), len(alpha)))


	for e in range(Total_num_experiments):
		All_Rwd_Lambda, All_Lambda_Alpha, All_Error_Lambda, All_Error_Lambda_Alpha = tree_backup_lambda(env, theta, num_episodes)

		Averaged_All_Rwd_Lambda = Averaged_All_Rwd_Lambda + All_Rwd_Lambda
		Averaged_All_Lambda_Alpha = Averaged_All_Lambda_Alpha + All_Lambda_Alpha
		Averaged_All_Error_Lambda = Averaged_All_Error_Lambda + All_Error_Lambda
		Averaged_All_Error_Lambda_Alpha = Averaged_All_Error_Lambda_Alpha + All_Error_Lambda_Alpha		

	Averaged_All_Rwd_Lambda = np.true_divide(Averaged_All_Rwd_Lambda, Total_num_experiments)
	Averaged_All_Lambda_Alpha = np.true_divide(Averaged_All_Lambda_Alpha, Total_num_experiments)
	Averaged_All_Error_Lambda = np.true_divide(Averaged_All_Error_Lambda, Total_num_experiments)
	Averaged_All_Error_Lambda_Alpha = np.true_divide(Averaged_All_Error_Lambda_Alpha, Total_num_experiments)		


	np.save('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Project_652/Code/Linear_Approximator/Eligibility_Traces/Accumulating_Traces/Cliff_Walking_Results/'  + 'Tree Backup(lambda)_RBF_' +  'Reward_Lambda_' + '.npy', Averaged_All_Rwd_Lambda)
	np.save('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Project_652/Code/Linear_Approximator/Eligibility_Traces/Accumulating_Traces/Cliff_Walking_Results/'  + 'Tree Backup(lambda)_RBF_' +  'Lambda_Alpha' + '.npy', Averaged_All_Lambda_Alpha)
	np.save('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Project_652/Code/Linear_Approximator/Eligibility_Traces/Accumulating_Traces/Cliff_Walking_Results/'  + 'Tree Backup(lambda)_RBF_' +  'Error_Lambda_' + '.npy', Averaged_All_Error_Lambda)
	np.save('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Project_652/Code/Linear_Approximator/Eligibility_Traces/Accumulating_Traces/Cliff_Walking_Results/'  + 'Tree Backup(lambda)_RBF_' +  'Error_Lambda_Alpha' + '.npy', Averaged_All_Error_Lambda_Alpha)

	# plotting.plot_episode_stats(stats_tree_lambda)
	env.close()



if __name__ == '__main__':
	main()


