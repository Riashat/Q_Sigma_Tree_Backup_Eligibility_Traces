import gym
import itertools
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import random


from collections import namedtuple
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict


from lib import plotting
from lib.envs.cliff_walking import CliffWalkingEnv
from lib import plotting
env = CliffWalkingEnv()


def make_epsilon_greedy_policy(Q, epsilon, nA):

	def policy_fn(observation):
		A = np.ones(nA, dtype=float) * epsilon/nA
		best_action = np.argmax(Q[observation])
		A[best_action] += ( 1.0 - epsilon)
		return A

	return policy_fn

def chosen_action(Q):
	best_action = np.argmax(Q)
	return best_action


def create_random_policy(nA):
    """
    Creates a random policy function.
    
    Args:
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes an observation as input and returns a vector
        of action probabilities
    """
    A = np.ones(nA, dtype=float) / nA
    def policy_fn(observation):
        return A
    return policy_fn




def behaviour_policy_epsilon_greedy(Q, epsilon, nA):

	def policy_fn(observation):
		A = np.ones(nA, dtype=float) * epsilon/nA
		best_action = np.argmax(Q[observation])
		A[best_action] += ( 1.0 - epsilon)
		return A

	return policy_fn

from numpy.random import binomial
def binomial_sigma(p):
	sample = binomial(n=1, p=p)
	return sample



"""
Online Tree Backup with Eligiblity Traces
"""
def tree_backup_lambda(env, num_episodes, discount_factor=1.0, alpha=0.1, epsilon=0.1):

	# Q = defaultdict(lambda : np.zeros(env.action_space.n))

	# stats = plotting.EpisodeStats(
	# 	episode_lengths=np.zeros(num_episodes),
	# 	episode_rewards=np.zeros(num_episodes),
	# 	episode_error=np.zeros(num_episodes))  


	lambda_param = np.array([0, 0.1, 0.15, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.975, 0.99, 1])

	alpha = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1])

	All_Rwd_Lambda = np.zeros(shape=(num_episodes, len(lambda_param)))
	All_Lambda_Alpha = np.zeros(shape=(len(lambda_param), len(alpha)))

	All_Error_Lambda = np.zeros(shape=(num_episodes, len(lambda_param)))	
	All_Error_Lambda_Alpha = np.zeros(shape=(len(lambda_param), len(alpha)))

	num_experiments = num_episodes


	for l in range(len(lambda_param)):

		print "Lambda Param", lambda_param[l]

		for alpha_param in range(len(alpha)):

			print "Alpha Param", alpha[alpha_param]

			Q = defaultdict(lambda : np.zeros(env.action_space.n))

			stats = plotting.EpisodeStats(
				episode_lengths=np.zeros(num_episodes),
				episode_rewards=np.zeros(num_episodes),
				episode_error=np.zeros(num_episodes))  

			for i_episode in range(num_episodes):

				print "Number of Episodes, Tree Backup (lambda)", i_episode

				policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
				state = env.reset()
				next_action = None

				action_probs = policy(state)
				action = np.random.choice(np.arange(len(action_probs)), p=action_probs)	


				eligibility = defaultdict(lambda : np.zeros(env.action_space.n))	
				#initialising eligibility traces

				for t in itertools.count():
					
					next_state, reward, done, _ = env.step(action)

					stats.episode_rewards[i_episode] += reward
					stats.episode_lengths[i_episode] = t

					next_action_probs = policy(next_state)
					next_action = np.random.choice(np.arange(len(next_action_probs)), p = next_action_probs)
					action_probability = next_action_probs[next_action]


					V = np.sum(next_action_probs * Q[next_state])

					Delta = reward + discount_factor * V - Q[state][action]

					rms_error = np.sqrt(np.sum((reward + discount_factor * V - Q[state][action])**2)/num_experiments)

					stats.episode_error[i_episode] += rms_error

					eligibility[state][action] = 1

					# Q[state][action] = Q[state][action] + alpha * Delta * eligibility[state][action]

					for s in range(env.observation_space.n):
						for a in range(env.action_space.n):
							Q[s][a] = Q[s][a] + alpha[alpha_param] * Delta * eligibility[s][a]

							eligibility[s][a] = eligibility[s][a] * discount_factor * lambda_param[l] * action_probability

					if done:
						break

					action = next_action
					state = next_state

			cum_rwd_per_episode = np.array([pd.Series(stats.episode_rewards).rolling(1, min_periods=1).mean()])
			cum_error_per_episode = np.array([pd.Series(stats.episode_error).rolling(1, min_periods=1).mean()])


			All_Rwd_Lambda[:, l] = cum_rwd_per_episode
			All_Error_Lambda[:, l] = cum_error_per_episode
			All_Lambda_Alpha[l, alpha_param] = cum_error_per_episode.T[-1]
			All_Error_Lambda_Alpha[l, alpha_param] = cum_error_per_episode.T[-1]


	return All_Rwd_Lambda, All_Lambda_Alpha, All_Error_Lambda, All_Error_Lambda_Alpha



def main():
	print "Tree Backup(lambda)"
	env = CliffWalkingEnv()
	Total_num_experiments = 10
	num_episodes = 1000

	lambda_param = np.array([0, 0.1, 0.15, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.975, 0.99, 1])
	alpha = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1])


	Averaged_All_Rwd_Lambda = np.zeros(shape=(num_episodes, len(lambda_param)))
	Averaged_All_Lambda_Alpha = np.zeros(shape=(len(lambda_param), len(alpha)))
	Averaged_All_Error_Lambda = np.zeros(shape=(num_episodes, len(lambda_param)))	
	Averaged_All_Error_Lambda_Alpha = np.zeros(shape=(len(lambda_param), len(alpha)))


	for e in range(Total_num_experiments):
		All_Rwd_Lambda, All_Lambda_Alpha, All_Error_Lambda, All_Error_Lambda_Alpha = tree_backup_lambda(env, num_episodes)

		Averaged_All_Rwd_Lambda = Averaged_All_Rwd_Lambda + All_Rwd_Lambda
		Averaged_All_Lambda_Alpha = Averaged_All_Lambda_Alpha + All_Lambda_Alpha
		Averaged_All_Error_Lambda = Averaged_All_Error_Lambda + All_Error_Lambda
		Averaged_All_Error_Lambda_Alpha = Averaged_All_Error_Lambda_Alpha + All_Error_Lambda_Alpha		

	Averaged_All_Rwd_Lambda = np.true_divide(Averaged_All_Rwd_Lambda, Total_num_experiments)
	Averaged_All_Lambda_Alpha = np.true_divide(Averaged_All_Lambda_Alpha, Total_num_experiments)
	Averaged_All_Error_Lambda = np.true_divide(Averaged_All_Error_Lambda, Total_num_experiments)
	Averaged_All_Error_Lambda_Alpha = np.true_divide(Averaged_All_Error_Lambda_Alpha, Total_num_experiments)		


	np.save('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Project_652/Code/Tabular/Tree_Backup_Results/'  + 'Tree Backup(lambda)_' +  'Reward_Lambda_' + '.npy', Averaged_All_Rwd_Lambda)
	np.save('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Project_652/Code/Tabular/Tree_Backup_Results/'  + 'Tree Backup(lambda)_' +  'Lambda_Alpha' + '.npy', Averaged_All_Lambda_Alpha)
	np.save('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Project_652/Code/Tabular/Tree_Backup_Results/'  + 'Tree Backup(lambda)_' +  'Error_Lambda_' + '.npy', Averaged_All_Error_Lambda)
	np.save('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Project_652/Code/Tabular/Tree_Backup_Results/'  + 'Tree Backup(lambda)_' +  'Error_Lambda_Alpha' + '.npy', Averaged_All_Error_Lambda_Alpha)

	# plotting.plot_episode_stats(stats_tree_lambda)
	env.close()



if __name__ == '__main__':
	main()


