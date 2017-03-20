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
def tree_backup_one_step(env, num_episodes, discount_factor=1.0, alpha=0.1, epsilon=0.1):

	Q = defaultdict(lambda : np.zeros(env.action_space.n))

	stats = plotting.EpisodeStats(
		episode_lengths=np.zeros(num_episodes),
		episode_rewards=np.zeros(num_episodes),
		episode_error=np.zeros(num_episodes))  


	lambda_param = np.array([1])

	alpha = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1])

	All_Rwd_Alpha = np.zeros(shape=(num_episodes, len(alpha)))
	All_Error_Alpha = np.zeros(shape=(num_episodes, len(alpha)))	


	num_experiments = num_episodes


	for l in range(len(lambda_param)):

		print "Lambda Param", lambda_param[l]

		for alpha_param in range(len(alpha)):

			print "Alpha Param", alpha[alpha_param]

			for i_episode in range(num_episodes):

				print "Number of Episodes, Tree Backup One Step", i_episode

				policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
				state = env.reset()
				next_action = None

				action_probs = policy(state)
				action = np.random.choice(np.arange(len(action_probs)), p=action_probs)	


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

					Q[state][action] += alpha[alpha_param] * Delta


					if done:
						break

					action = next_action
					state = next_state

			cum_rwd_per_episode = np.array([pd.Series(stats.episode_rewards).rolling(1, min_periods=1).mean()])
			cum_error_per_episode = np.array([pd.Series(stats.episode_error).rolling(1, min_periods=1).mean()])


			All_Rwd_Alpha[:, alpha_param] = cum_rwd_per_episode
			All_Error_Alpha[:, alpha_param] = cum_error_per_episode

	return All_Rwd_Alpha, All_Error_Alpha



def main():
	print "Tree Backup One Step"
	env = CliffWalkingEnv()
	Total_num_experiments = 10
	num_episodes = 2000

	alpha = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1])

	Averaged_All_Rwd_Alpha = np.zeros(shape=(num_episodes, len(alpha)))
	Averaged_All_Error_Alpha = np.zeros(shape=(num_episodes, len(alpha)))	



	for e in range(Total_num_experiments):
		All_Rwd_Alpha, All_Error_Alpha = tree_backup_one_step(env, num_episodes)

		Averaged_All_Rwd_Alpha = Averaged_All_Rwd_Alpha + All_Rwd_Alpha
		Averaged_All_Error_Alpha = Averaged_All_Error_Alpha + All_Error_Alpha


	Averaged_All_Rwd_Alpha = np.true_divide(Averaged_All_Rwd_Alpha, Total_num_experiments)
	Averaged_All_Error_Alpha = np.true_divide(Averaged_All_Error_Alpha, Total_num_experiments)


	np.save('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Project_652/Code/Tabular/Tree_Backup_Results/'  + 'Tree Backup_One_Step_' +  'Reward_Alpha_' + '.npy', Averaged_All_Rwd_Alpha)
	np.save('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Project_652/Code/Tabular/Tree_Backup_Results/'  + 'Tree Backup_One_Step_' +  'Error_Alpha_' + '.npy', Averaged_All_Error_Alpha)

	env.close()



if __name__ == '__main__':
	main()


