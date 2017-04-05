import sys
#sys.path.insert(0, "/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Project_652/Code/Linear_Approximator/Eligibility_Traces/Accumulating_Traces/")
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




def q_sigma_lambda_on_policy_dynamic_sigma(env, theta, num_episodes,alpha, lambda_param, discount_factor=1.0, epsilon=0.1, epsilon_decay=1.0 ):
	

	
	stats = plotting.EpisodeStats(
		episode_lengths=np.zeros(num_episodes),
		episode_rewards=np.zeros(num_episodes))
	cumulative_errors = np.zeros(shape=(num_episodes, 1))
	sigma = 1
	sigma_decay = 0.995


	for i_episode in range(num_episodes):
		state_count=np.zeros(shape=(env.observation_space.n,1))

		print ("Episode Number, Q (sigma, lambda) On Policy, Dynamic Sigma:", i_episode)

		policy = epsilon_greedy_policy(theta, epsilon * epsilon_decay**i_episode, env.action_space.n)
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

			next_action_probs = policy(next_state)
			next_action = np.random.choice(np.arange(len(next_action_probs)), p = next_action_probs)

			next_features_state = featurize_state(next_state)
			next_q_values = np.dot(theta.T, next_features_state)
			next_q_values_state_action = next_q_values[next_action]

			V = np.sum(next_action_probs * next_q_values)

			if state_count[state]>=5:
				sigma = 1
			else:
				sigma=0


			Sigma_Effect = sigma * next_q_values_state_action + (1 - sigma) * V			

			td_target = reward + discount_factor * Sigma_Effect
			td_delta = td_target - q_values_state_action


			rms_error = np.sqrt(np.sum((td_delta)**2))
			cumulative_errors[i_episode, :] += rms_error

			next_action_probability = next_action_probs[next_action]


			eligibility[:, action] = eligibility[:, action] + features_state


			for s in range(features_state.shape[0]):
				for a in range(env.action_space.n):
					#eligibility[s, a] = eligibility[s, a] * discount_factor * lambda_param * next_action_probability
					eligibility[s, a] = ( 1 - sigma ) * eligibility[s, a] * discount_factor * lambda_param * next_action_probability    +     sigma * discount_factor * lambda_param * eligibility[s, a]					


			theta[:, action] += alpha * eligibility[:, action] * td_delta

			if done:
				
				break

			state = next_state
			
	



	return stats, cumulative_errors


def main():

	print ("Q_Sigma_Lambda_Offpolicy_Dynamic_sigma")

	theta = np.zeros(shape=(400, env.action_space.n))
	num_experiments=50

	num_episodes =1000
	smoothing_window = 1
	lambda_initialised=np.array([1,0.99,0.975,0.95,0.9,0.8,0.4]) 
	alpha_initialised=np.array([0.2,0.4,0.6,0.8,1])
	lambda_reward=np.zeros(shape=(num_episodes,len(lambda_initialised)))
	lambda_error=np.zeros(shape=(num_episodes,len(lambda_initialised)))
	alpha_reward=np.zeros(shape=(num_episodes,len(alpha_initialised)))
	alpha_error=np.zeros(shape=(num_episodes,len(alpha_initialised)))
	lambda_alpha_reward=np.zeros(shape=(len(alpha_initialised),len(lambda_initialised)))
	lambda_alpha_error=np.zeros(shape=(len(alpha_initialised),len(lambda_initialised)))

	for i in range(len(lambda_initialised)):
		for j in range(len(alpha_initialised)):
			lambda_param=lambda_initialised[i]
			alpha=alpha_initialised[j]
			print ("Q_Sigma_Lambda_Offpolicy_Static_sigma")
			stats,cum_error=q_sigma_lambda_on_policy_dynamic_sigma(env,theta,num_episodes,alpha,lambda_param)
			alpha_reward[:,j]=stats.episode_rewards
			alpha_error[:,j]=cum_error.T
			lambda_reward[:,i]=stats.episode_rewards
			lambda_error[:,i]=cum_error.T
			lambda_alpha_reward[i,j]=stats.episode_rewards[-1]
			lambda_alpha_error[i,j]=cum_error[-1]
			
			np.save('/home/raihan/Desktop/Final_Project_Codes/Windy_GridWorld/Experimental_Results /finetuning/'  + 'QsigmaLambdaOnPolicy_dynamic_lambda_alpha_reward' + '.npy',lambda_alpha_reward)
			np.save('/home/raihan/Desktop/Final_Project_Codes/Windy_GridWorld/Experimental_Results /finetuning/'  + 'QsigmaLambdaOnPolicy_dynamic_lambda_alpha_reward' + '.npy',lambda_alpha_error)

	
	env.close()



	


if __name__ == '__main__':
	main()



