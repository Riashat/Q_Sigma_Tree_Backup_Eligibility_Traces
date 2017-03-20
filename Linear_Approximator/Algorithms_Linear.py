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


"""
Environment
"""

from collections import defaultdict
from lib.envs.windy_gridworld import WindyGridworldEnv
from lib import plotting
env = WindyGridworldEnv()




"""
Feature Extactor
"""
observation_examples = np.array([env.observation_space.sample() for x in range(1)])
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(observation_examples)


#convert states to a feature representation:
#used an RBF sampler here for the feature map
featurizer = sklearn.pipeline.FeatureUnion([
        ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
        ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
        ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
        ("rbf4", RBFSampler(gamma=0.5, n_components=100))
        ])
featurizer.fit(scaler.transform(observation_examples))



class Estimator():

	"""
	Class to define the value function - Linear Function Approximator in this case
	"""
	def __init__(self):
		self.models = []

		for _ in range(env.action_space.n):
			model = SGDRegressor(learning_rate = "constant")
			model.partial_fit([self.featurize_state(env.reset())], [0])
			self.models.append(model)

	def featurize_state(self, state):
		state = np.array([state])
		scaled = scaler.transform([state])
		featurized = featurizer.transform(scaled)
		return featurized[0]



	def predict(self, s, a=None):
		features = self.featurize_state(s)
		if not a:
			return np.array([m.predict([features])[0] for m in self.models])
		else:
			return self.models[a].predict([features])[0]


	def predict_s_a(self, s, a=None):
		features = self.featurize_state_action(s, a)
		if not a:
			return np.array([m.predict([features])[0] for m in self.models])
		else:
			return self.models[a].predict([features])[0]


	def update(self, s, a, target):

		#updates the estimator parameters for given s,a towards target y
		features = self.featurize_state(s)
		self.models[a].partial_fit([features], [target])


"""
Agent policies
"""

def make_epsilon_greedy_policy(estimator, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.
    
    Args:
        estimator: An estimator that returns q values for a given state
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    
    """
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(observation)
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn




"""
Main Baselines
"""
def q_learning(env, estimator, num_episodes, discount_factor=1.0, epsilon=0.1, epsilon_decay=0.999):

	#q-learning algorithm with linear function approximation here

	#estimator : Estimator of Q^w(s,a)	- function approximator
	stats = plotting.EpisodeStats(
		episode_lengths=np.zeros(num_episodes),
		episode_rewards=np.zeros(num_episodes))  

	for i_episode in range(num_episodes):
		print "Episode Number, Q Learning:", i_episode
		#agent policy based on the greedy maximisation of Q
		policy = make_epsilon_greedy_policy(estimator, epsilon * epsilon_decay**i_episode, env.action_space.n)
		last_reward = stats.episode_rewards[i_episode - 1]
		state = env.reset()

		next_action = None

		#for each one step in the environment
		for t in itertools.count():
			if next_action is None:
				action_probs = policy(state)
				action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
			else:
				action = next_action

			next_state, reward, done, _ = env.step(action)
			stats.episode_rewards[i_episode] += reward
			stats.episode_lengths[i_episode] = t

			#update Q-values for the next state
			q_values_next = estimator.predict(next_state)

			#Q-value TD Target
			td_target = reward + discount_factor * np.max(q_values_next)

			#update the Q values
			#not this anymore
			#Q[state][action] += alpha * td_delta
			estimator.update(state, action, td_target)
			if done:
				break
			state = next_state
	return stats



def sarsa(env, estimator, num_episodes, discount_factor=1.0, epsilon=0.1, epsilon_decay=1.0):
	#estimator : Estimator of Q^w(s,a)	- function approximator
	stats = plotting.EpisodeStats(episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes))

	for i_episode in range(num_episodes):
		print "Episode Number, SARSA:", i_episode
		policy = make_epsilon_greedy_policy(estimator, epsilon * epsilon_decay**i_episode, env.action_space.n)
		last_reward = stats.episode_rewards[i_episode - 1]
		state = env.reset()
		next_action = None

		action_probs = policy(state)
		action = np.random.choice(np.arange(len(action_probs)), p=action_probs)		

		# if next_action is None:
		# 	action_probs = policy(state)
		# 	action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
		# else:
		# 	action = next_action

		# action_probs = policy(state)
		# action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

		for t in itertools.count():

			next_state, reward, done, _ = env.step(action)

			stats.episode_rewards[i_episode] += reward
			stats.episode_lengths[i_episode] = t

			next_action_probs = policy(next_state)
			next_action = np.random.choice(np.arange(len(next_action_probs)), p = next_action_probs)

			#update Q-values for the next state, next action
			q_values_next = estimator.predict(next_state)

			q_next_state_next_action = q_values_next[next_action] 

			td_target = reward + discount_factor * q_next_state_next_action

			estimator.update(state, action, td_target)

			if done:
				break

			state = next_state
			action = next_action

	return stats


"""
TREE BACKUP ALGORITHMS
"""


"""
On Policy Expected SARSA - One Step Tree Backup
"""
def on_policy_expected_sarsa(env, estimator, num_episodes, discount_factor=1.0, epsilon=0.1, epsilon_decay=0.999):

	#q-learning algorithm with linear function approximation here

	#estimator : Estimator of Q^w(s,a)	- function approximator
	stats = plotting.EpisodeStats(
		episode_lengths=np.zeros(num_episodes),
		episode_rewards=np.zeros(num_episodes))  

	for i_episode in range(num_episodes):
		print "Episode Number, Expected SARSA:", i_episode
		#agent policy based on the greedy maximisation of Q
		policy = make_epsilon_greedy_policy(estimator, epsilon * epsilon_decay**i_episode, env.action_space.n)
		last_reward = stats.episode_rewards[i_episode - 1]
		state = env.reset()

		next_action = None

		#for each one step in the environment
		for t in itertools.count():

			
			if next_action is None:
				action_probs = policy(state)
				action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
			else:
				action = next_action

			next_state, reward, done, _ = env.step(action)
			stats.episode_rewards[i_episode] += reward
			stats.episode_lengths[i_episode] = t

			#update Q-values for the next state
			q_values_next = estimator.predict(next_state)

			next_action_probs = policy(next_state)
			next_action = np.random.choice(np.arange(len(next_action_probs)), p = next_action_probs)

			V = np.sum( next_action_probs * q_values_next)


			#Q-value TD Target
			td_target = reward + discount_factor * V

			estimator.update(state, action, td_target)
			if done:
				break
			state = next_state
	return stats





def two_step_tree_backup(env, estimator, num_episodes, discount_factor=1.0, epsilon=0.1, epsilon_decay=0.999):

	#q-learning algorithm with linear function approximation here

	#estimator : Estimator of Q^w(s,a)	- function approximator
	stats = plotting.EpisodeStats(
		episode_lengths=np.zeros(num_episodes),
		episode_rewards=np.zeros(num_episodes))  

	for i_episode in range(num_episodes):
		print "Episode Number, Two Step Tree Backup:", i_episode
		#agent policy based on the greedy maximisation of Q
		policy = make_epsilon_greedy_policy(estimator, epsilon * epsilon_decay**i_episode, env.action_space.n)
		last_reward = stats.episode_rewards[i_episode - 1]
		state = env.reset()

		next_action = None

		#for each one step in the environment
		for t in itertools.count():
			if next_action is None:
				action_probs = policy(state)
				action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
			else:
				action = next_action

			next_state, reward, done, _ = env.step(action)
			if done:
				break
				
			stats.episode_rewards[i_episode] += reward
			stats.episode_lengths[i_episode] = t

			#update Q-values for the next state
			q_values_next = estimator.predict(next_state)

			next_action_probs = policy(next_state)
			next_action = np.random.choice(np.arange(len(next_action_probs)), p = next_action_probs)

			V = np.sum( next_action_probs * q_values_next)


			next_next_state, next_reward, _, _ = env.step(next_action)
			next_next_action_probs = policy(next_next_state)
			next_next_action = np.random.choice(np.arange(len(next_next_action_probs)), p = next_next_action_probs)

			q_values_next_next = estimator.predict(next_next_state)

			next_V = np.sum(next_next_action_probs * q_values_next_next)

			q_next_next_state_next_next_action = q_values_next_next[next_next_action]

			Delta = next_reward + discount_factor * next_V - q_next_next_state_next_next_action

			next_action_selection_probability = np.max(next_action_probs)

			td_target = reward + discount_factor * V + discount_factor * next_action_selection_probability * Delta


			estimator.update(state, action, td_target)

			# if done:
			# 	break
			state = next_state

	return stats




def three_step_tree_backup(env, estimator, num_episodes, discount_factor=1.0, epsilon=0.1, epsilon_decay=0.999):

	#q-learning algorithm with linear function approximation here

	#estimator : Estimator of Q^w(s,a)	- function approximator
	stats = plotting.EpisodeStats(
		episode_lengths=np.zeros(num_episodes),
		episode_rewards=np.zeros(num_episodes))  

	for i_episode in range(num_episodes):
		print "Episode Number, Three Step Tree Backup:", i_episode
		#agent policy based on the greedy maximisation of Q
		policy = make_epsilon_greedy_policy(estimator, epsilon * epsilon_decay**i_episode, env.action_space.n)
		last_reward = stats.episode_rewards[i_episode - 1]
		state = env.reset()

		next_action = None

		#for each one step in the environment
		for t in itertools.count():
			if next_action is None:
				action_probs = policy(state)
				action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
			else:
				action = next_action

			next_state, reward, done, _ = env.step(action)
			if done:
				break
			

			stats.episode_rewards[i_episode] += reward
			stats.episode_lengths[i_episode] = t

			q_values = estimator.predict(state)
			q_values_state_action = q_values[action]


			next_action_probs = policy(next_state)
			next_action = np.random.choice(np.arange(len(next_action_probs)), p = next_action_probs)
			

			#update Q-values for the next state
			q_values_next = estimator.predict(next_state)
			q_values_next_state_next_action = q_values_next[next_action]

			V = np.sum( next_action_probs * q_values_next)

			Delta = reward + discount_factor * V - q_values_state_action


			next_next_state, next_reward, _, _ = env.step(next_action)
			next_next_action_probs = policy(next_next_state)
			next_next_action = np.random.choice(np.arange(len(next_next_action_probs)), p = next_next_action_probs)

			q_values_next_next = estimator.predict(next_next_state)
			q_values_next_next_state_next_next_action = q_values_next_next[next_next_action]

			next_V = np.sum(next_next_action_probs * q_values_next_next)

			Delta_t_1 = next_reward + discount_factor * next_V - q_values_next_state_next_action


			next_next_next_state, next_next_reward, _, _ = env.step(next_next_action)
			next_next_next_action_probs = policy(next_next_next_state)
			next_next_next_action = np.random.choice(np.arange(len(next_next_next_action_probs)), p = next_next_next_action_probs)

			q_values_next_next_next = estimator.predict(next_next_next_state)
			q_values_next_next_next_state_next_next_next_action = q_values_next_next_next[next_next_next_action]

			next_next_V = np.sum(next_next_next_action_probs * q_values_next_next_next)

			Delta_t_2 = next_next_reward + discount_factor * next_next_V - q_values_next_next_state_next_next_action


			next_action_selection_probability = np.max(next_action_probs)
			next_next_action_selection_probability = np.max(next_next_action_probs)


			td_target = q_values_state_action + Delta + discount_factor * next_action_selection_probability * Delta_t_1 + discount_factor * discount_factor * next_action_selection_probability * next_next_action_selection_probability * Delta_t_2


			estimator.update(state, action, td_target)

			# if done:
			# 	break
			state = next_state

	return stats





"""
Q(sigma) Algorithms
"""


"""
On Policy Q Sigma Algorithm with Linear Function Approximator
"""
def Q_Sigma_On_Policy(env, estimator, num_episodes, discount_factor=1.0, epsilon=0.1, epsilon_decay=0.999):

	#q-learning algorithm with linear function approximation here

	#estimator : Estimator of Q^w(s,a)	- function approximator
	stats = plotting.EpisodeStats(
		episode_lengths=np.zeros(num_episodes),
		episode_rewards=np.zeros(num_episodes))  

	for i_episode in range(num_episodes):
		print "Episode Number, Q(sigma):", i_episode
		#agent policy based on the greedy maximisation of Q
		policy = make_epsilon_greedy_policy(estimator, epsilon * epsilon_decay**i_episode, env.action_space.n)

		state = env.reset()
		next_action = None

		#for each one step in the environment
		for t in itertools.count():
			if next_action is None:
				action_probs = policy(state)
				action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
			else:
				action = next_action


			next_state, reward, done, _ = env.step(action)
			if done:
				break
			
			stats.episode_rewards[i_episode] += reward
			stats.episode_lengths[i_episode] = t


			#update Q-values for the next state
			q_values_next = estimator.predict(next_state)

			next_action_probs = policy(next_state)
			next_action = np.random.choice(np.arange(len(next_action_probs)), p = next_action_probs)

			sigma = random.randint(0,1)

			V = np.sum( next_action_probs * q_values_next)

			q_next_state_next_action = q_values_next[next_action]

			Sigma_Effect = sigma * q_next_state_next_action+ (1 - sigma) * V

			td_target = reward + discount_factor * Sigma_Effect

			#Q-value TD Target
			td_target = reward + discount_factor * V

			estimator.update(state, action, td_target)
			if done:
				break
			state = next_state
	return stats




def featurize_state(state):

	state = np.array([state])

	scaled = scaler.transform([state])
	featurized = featurizer.transform(scaled)
	return featurized[0]




# def make_epsilon_greedy_policy(theta, epsilon, nA):
#     def policy_fn(observation):
#         A = np.ones(nA, dtype=float) * epsilon / nA
#         phi = featurize_state(observation)
#         q_values = np.dot(theta.T, phi)
#         best_action = np.argmax(q_values)
#         A[best_action] += (1.0 - epsilon)
#         return A
#     return policy_fn

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



def behaviour_policy_Boltzmann(theta, tau, nA):

	def policy_fn(observation):
		phi = featurize_state(observation)
		q_values = np.dot(theta.T, phi)
		exp_tau = q_values / tau
		policy = np.exp(exp_tau) / np.sum(np.exp(exp_tau), axis=0)
		A = policy
		return A
	return policy_fn




from numpy.random import binomial
def binomial_sigma(p):
	sample = binomial(n=1, p=p)
	return sample


def epsilon_greedy_policy(theta, epsilon, nA):
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        phi = featurize_state(observation)
        q_values = np.dot(theta.T, phi)
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn



def Q_Sigma_Off_Policy(env, theta, num_episodes, discount_factor=1.0, epsilon=0.1, epsilon_decay=0.999):

	#q-learning algorithm with linear function approximation here

	#estimator : Estimator of Q^w(s,a)	- function approximator
	stats = plotting.EpisodeStats(
		episode_lengths=np.zeros(num_episodes),
		episode_rewards=np.zeros(num_episodes))  

	alpha = 0.01
	tau = 1

	for i_episode in range(num_episodes):

		print "Epsisode Number Off Policy Q(sigma)", i_episode

		off_policy = behaviour_policy_Boltzmann(theta, tau * epsilon_decay**i_episode, env.action_space.n)
		policy = epsilon_greedy_policy(theta, epsilon * epsilon_decay**i_episode, env.action_space.n)

		state = env.reset()
		next_action = None


		for t in itertools.count():

			if next_action is None:
				action_probs = off_policy(state)
				action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
			else:
				action = next_action

			state_t_1, reward, done, _ = env.step(action)

			if done:
				break			

			stats.episode_rewards[i_episode] += reward
			stats.episode_lengths[i_episode] = t



			# q_values = estimator.predict(state)
			# q_values_state_action = q_values[action]
			#evaluate Q(current state, current action)
			features_state = featurize_state(state)
			q_values = np.dot(theta.T, features_state)
			q_values_state_action = q_values[action]



			#select sigma value
			probability = 0.5
			sigma_t_1 = binomial_sigma(probability)

			#select next action based on the behaviour policy at next state
			next_action_probs = off_policy(state_t_1)
			action_t_1 = np.random.choice(np.arange(len(next_action_probs)), p = next_action_probs)


			# q_values_t_1 = estimator.predict(state_t_1)
			# q_values_next_state_next_action = q_values_t_1[action_t_1]
			features_state_1 = featurize_state(state_t_1)
			q_values_t_1 = np.dot(theta.T, features_state_1)
			q_values_next_state_next_action = q_values_t_1[action_t_1]


			on_policy_next_action_probs = policy(state_t_1)
			on_policy_a_t_1 = np.random.choice(np.arange(len(on_policy_next_action_probs)), p = on_policy_next_action_probs)
			V_t_1 = np.sum( on_policy_next_action_probs * q_values_t_1 )

			Delta_t = reward + discount_factor * ( sigma_t_1 * q_values_next_state_next_action + (1 - sigma_t_1) * V_t_1  ) - q_values_state_action


			"""
			target for one step
			1 step TD Target --- G_t(1)
			"""
			td_target = q_values_state_action + Delta_t 

			td_error = td_target -  q_values_state_action 

			# estimator.update(state, action, new_td_target)
			theta[:, action] += alpha * td_error * features_state


			state = state_t_1

	return stats





def Q_Sigma_Off_Policy_2_Step(env, theta, num_episodes, discount_factor=1.0, epsilon=0.1, epsilon_decay=0.999):

	#q-learning algorithm with linear function approximation here

	#estimator : Estimator of Q^w(s,a)	- function approximator
	stats = plotting.EpisodeStats(
		episode_lengths=np.zeros(num_episodes),
		episode_rewards=np.zeros(num_episodes))  
	alpha = 0.01

	tau = 1
	tau_decay = 0.999

	sigma = 1
	sigma_decay = 0.995

	for i_episode in range(num_episodes):

		print "Epsisode Number Off Policy Q(sigma) 2 Step", i_episode



		off_policy = behaviour_policy_Boltzmann(theta, tau * epsilon_decay**i_episode, env.action_space.n)
		policy = epsilon_greedy_policy(theta, epsilon * epsilon_decay**i_episode, env.action_space.n)

		tau = tau * tau_decay

		if tau < 0.0001:
			tau = 0.0001

		state = env.reset()
		next_action = None

		for t in itertools.count():

			if next_action is None:
				action_probs = off_policy(state)
				action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
			else:
				action = next_action

			state_t_1, reward, done, _ = env.step(action)

			stats.episode_rewards[i_episode] += reward
			stats.episode_lengths[i_episode] = t

			if done:
				sigma = sigma * sigma_decay
				if sigma < 0.0001:
					sigma = 0.0001	
				break			

			# q_values = estimator.predict(state)
			# q_values_state_action = q_values[action]
			#evaluate Q(current state, current action)
			features_state = featurize_state(state)
			q_values = np.dot(theta.T, features_state)
			q_values_state_action = q_values[action]


			#select sigma value
			# probability = 0.5
			sigma_t_1 = sigma

			#select next action based on the behaviour policy at next state
			next_action_probs = off_policy(state_t_1)
			action_t_1 = np.random.choice(np.arange(len(next_action_probs)), p = next_action_probs)


			# q_values_t_1 = estimator.predict(state_t_1)
			# q_values_next_state_next_action = q_values_t_1[action_t_1]
			features_state_1 = featurize_state(state_t_1)
			q_values_t_1 = np.dot(theta.T, features_state_1)
			q_values_next_state_next_action = q_values_t_1[action_t_1]


			on_policy_next_action_probs = policy(state_t_1)
			on_policy_a_t_1 = np.random.choice(np.arange(len(on_policy_next_action_probs)), p = on_policy_next_action_probs)
			V_t_1 = np.sum( on_policy_next_action_probs * q_values_t_1 )

			Delta_t = reward + discount_factor * ( sigma_t_1 * q_values_next_state_next_action + (1 - sigma_t_1) * V_t_1  ) - q_values_state_action



			state_t_2, next_reward, _, _ = env.step(action_t_1)

			next_next_action_probs = off_policy(state_t_2)
			action_t_2 = np.random.choice(np.arange(len(next_next_action_probs)), p = next_next_action_probs)


			# q_values_t_2 = estimator.predict(state_t_2)
			# q_values_next_next_state_next_next_action = q_values_t_2[action_t_2]
			features_state_2 = featurize_state(state_t_2)
			q_values_t_2 = np.dot(theta.T, features_state_2)
			q_values_next_next_state_next_next_action = q_values_t_2[action_t_2]




			on_policy_next_next_action_probs = policy(state_t_2)
			on_policy_a_t_2 = np.random.choice(np.arange(len(on_policy_next_next_action_probs)), p = on_policy_next_next_action_probs)
			V_t_2 = np.sum( on_policy_next_next_action_probs * q_values_t_2  )
			
			sigma_t_2 = sigma

			Delta_t_1 = next_reward + discount_factor * (  sigma_t_2 * q_values_next_next_state_next_next_action + (1 - sigma_t_2) * V_t_2   ) - q_values_next_state_next_action

			"""
			2 step TD Target --- G_t(2)
			"""
			on_policy_action_probability = on_policy_next_action_probs[on_policy_a_t_1]
			off_policy_action_probability = next_action_probs[action_t_1]

			td_target = q_values_state_action + Delta_t + discount_factor * ( (1 - sigma_t_1) *  on_policy_action_probability + sigma_t_1 ) * Delta_t_1

			"""
			Computing Importance Sampling Ratio
			"""
			rho = np.divide( on_policy_action_probability, off_policy_action_probability )
			rho_sigma = sigma_t_1 * rho + 1 - sigma_t_1

			td_error = td_target -  q_values_state_action 

			# estimator.update(state, action, new_td_target)
			theta[:, action] += alpha * rho_sigma * td_error * features_state

			state = state_t_1
			
	return stats





def Q_Sigma_Off_Policy_3_Step(env, theta, num_episodes, discount_factor=1.0, epsilon=0.1, epsilon_decay=0.999):

	#q-learning algorithm with linear function approximation here

	#estimator : Estimator of Q^w(s,a)	- function approximator
	stats = plotting.EpisodeStats(
		episode_lengths=np.zeros(num_episodes),
		episode_rewards=np.zeros(num_episodes))  

	alpha = 0.01
	alpha = 0.01

	tau = 1
	tau_decay = 0.999

	sigma = 1
	sigma_decay = 0.995



	for i_episode in range(num_episodes):

		print "Epsisode Number Off Policy Q(sigma) 3 Step", i_episode

		off_policy = behaviour_policy_Boltzmann(theta, tau * epsilon_decay**i_episode, env.action_space.n)
		policy = epsilon_greedy_policy(theta, epsilon * epsilon_decay**i_episode, env.action_space.n)

		tau = tau * tau_decay

		if tau < 0.0001:
			tau = 0.0001


		state = env.reset()

		next_action = None


		for t in itertools.count():

			if next_action is None:
				action_probs = off_policy(state)
				action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
			else:
				action = next_action

			state_t_1, reward, done, _ = env.step(action)

			stats.episode_rewards[i_episode] += reward
			stats.episode_lengths[i_episode] = t

			if done:
				sigma = sigma * sigma_decay
				if sigma < 0.0001:
					sigma = 0.0001	

				break			


			# q_values = estimator.predict(state)
			# q_values_state_action = q_values[action]
			#evaluate Q(current state, current action)
			features_state = featurize_state(state)
			q_values = np.dot(theta.T, features_state)
			q_values_state_action = q_values[action]


			#select sigma value
			probability = 0.5
			sigma_t_1 = binomial_sigma(probability)

			#select next action based on the behaviour policy at next state
			next_action_probs = off_policy(state_t_1)
			action_t_1 = np.random.choice(np.arange(len(next_action_probs)), p = next_action_probs)


			# q_values_t_1 = estimator.predict(state_t_1)
			# q_values_next_state_next_action = q_values_t_1[action_t_1]
			features_state_1 = featurize_state(state_t_1)
			q_values_t_1 = np.dot(theta.T, features_state_1)
			q_values_next_state_next_action = q_values_t_1[action_t_1]


			on_policy_next_action_probs = policy(state_t_1)
			on_policy_a_t_1 = np.random.choice(np.arange(len(on_policy_next_action_probs)), p = on_policy_next_action_probs)
			V_t_1 = np.sum( on_policy_next_action_probs * q_values_t_1 )

			Delta_t = reward + discount_factor * ( sigma_t_1 * q_values_next_state_next_action + (1 - sigma_t_1) * V_t_1  ) - q_values_state_action



			state_t_2, next_reward, _, _ = env.step(action_t_1)
			# if done:
			# 	break

			next_next_action_probs = off_policy(state_t_2)
			action_t_2 = np.random.choice(np.arange(len(next_next_action_probs)), p = next_next_action_probs)


			# q_values_t_2 = estimator.predict(state_t_2)
			# q_values_next_next_state_next_next_action = q_values_t_2[action_t_2]
			features_state_2 = featurize_state(state_t_2)
			q_values_t_2 = np.dot(theta.T, features_state_2)
			q_values_next_next_state_next_next_action = q_values_t_2[action_t_2]




			on_policy_next_next_action_probs = policy(state_t_2)
			on_policy_a_t_2 = np.random.choice(np.arange(len(on_policy_next_next_action_probs)), p = on_policy_next_next_action_probs)
			V_t_2 = np.sum( on_policy_next_next_action_probs * q_values_t_2  )
			
			sigma_t_2 = binomial_sigma(probability)



			Delta_t_1 = next_reward + discount_factor * (  sigma_t_2 * q_values_next_next_state_next_next_action + (1 - sigma_t_2) * V_t_2   ) - q_values_next_state_next_action


			"""
			3 step TD Target --- G_t(2)
			"""
			state_t_3, next_next_reward, _, _ = env.step(action_t_2)
			# if done:
			# 	break


			next_next_next_action_probs = off_policy(state_t_3)
			action_t_3 = np.random.choice(np.arange(len(next_next_next_action_probs)), p = next_next_next_action_probs)

			features_state_3 = featurize_state(state_t_3)
			q_values_t_3 = np.dot(theta.T,features_state_3)
			q_values_next_next_next_state_next_next_next_action = q_values_t_3[action_t_3]

			on_policy_next_next_next_action_probs = policy(state_t_3)
			on_policy_a_t_3 = np.random.choice(np.arange(len(on_policy_next_next_next_action_probs)), p = on_policy_next_next_next_action_probs)
			V_t_3 = np.sum(on_policy_next_next_next_action_probs * q_values_t_3)

			sigma_t_3 = sigma

			Delta_t_2 = next_next_reward + discount_factor * (sigma_t_3 * q_values_next_next_next_state_next_next_next_action + (1 - sigma_t_3) * V_t_3 ) -  q_values_next_next_state_next_next_action



			on_policy_action_probability = on_policy_next_action_probs[on_policy_a_t_1]
			off_policy_action_probability = next_action_probs[action_t_1]

			on_policy_next_action_probability = on_policy_next_next_action_probs[on_policy_a_t_2]
			off_policy_next_action_probability = next_next_action_probs[action_t_2]



			td_target = q_values_state_action + Delta_t + discount_factor * ( (1 - sigma_t_1) *  on_policy_action_probability + sigma_t_1 ) * Delta_t_1 + discount_factor * ( (1 - sigma_t_2)  * on_policy_next_action_probability + sigma_t_2 ) * Delta_t_2

			"""
			Computing Importance Sampling Ratio
			"""
			rho = np.divide( on_policy_action_probability, off_policy_action_probability )
			rho_1 = np.divide( on_policy_next_action_probability, off_policy_next_action_probability )

			rho_sigma = sigma_t_1 * rho + 1 - sigma_t_1
			rho_sigma_1 = sigma_t_2 * rho_1 + 1 - sigma_t_2

			all_rho_sigma = rho_sigma * rho_sigma_1

			td_error = td_target -  q_values_state_action 

			# estimator.update(state, action, new_td_target)
			theta[:, action] += alpha * all_rho_sigma * td_error * features_state

			state = state_t_1
			
	return stats





def main():

	# print "Q Learning"
	# estimator = Estimator()
	# num_episodes = 1000
	# smoothing_window = 1
	# stats_q_learning = q_learning(env, estimator, num_episodes, epsilon=0.1)
	# rewards_smoothed_stats_q_learning = pd.Series(stats_q_learning.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()	
	# cum_rwd = rewards_smoothed_stats_q_learning
	# np.save('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Project_652/Code/Linear_Approximator/WindyGridWorld_Results/'  + 'Q_Learning' + '.npy', cum_rwd)
	# env.close()


	# print "SARSA"
	# estimator = Estimator()
	# num_episodes = 1000
	# smoothing_window = 1
	# stats_sarsa = sarsa(env, estimator, num_episodes, epsilon=0.1)
	# rewards_smoothed_stats_sarsa = pd.Series(stats_sarsa.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()	
	# cum_rwd = rewards_smoothed_stats_sarsa
	# np.save('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Project_652/Code/Linear_Approximator/WindyGridWorld_Results/'  + 'SARSA' + '.npy', cum_rwd)
	# env.close()


	# print "On Policy Expected SARSA"
	# estimator = Estimator()
	# num_episodes = 1000
	# smoothing_window = 1
	# stats_expected_sarsa = on_policy_expected_sarsa(env, estimator, num_episodes, epsilon=0.1)
	# rewards_smoothed_stats_expected_sarsa = pd.Series(stats_expected_sarsa.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()	
	# cum_rwd = rewards_smoothed_stats_expected_sarsa
	# np.save('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Project_652/Code/Linear_Approximator/WindyGridWorld_Results/'  + 'Expected_SARSA' + '.npy', cum_rwd)
	# env.close()


	# print "Two Step Tree Backup"
	# estimator = Estimator()
	# num_episodes = 1000
	# smoothing_window = 1
	# stats_two_step_tree_backup = two_step_tree_backup(env, estimator, num_episodes, epsilon=0.1)
	# rewards_smoothed_stats_two_step_tree_backup = pd.Series(stats_two_step_tree_backup.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()	
	# cum_rwd = rewards_smoothed_stats_two_step_tree_backup
	# np.save('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Project_652/Code/Linear_Approximator/WindyGridWorld_Results/'  + 'Two_Step_Tree_Backup' + '.npy', cum_rwd)
	# env.close()


	# print "Three Step Tree Backup"
	# estimator = Estimator()
	# num_episodes = 1000
	# smoothing_window = 1
	# stats_three_step_tree_backup = three_step_tree_backup(env, estimator, num_episodes, epsilon=0.1)
	# rewards_smoothed_stats_three_step_tree_backup = pd.Series(stats_three_step_tree_backup.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()	
	# cum_rwd = rewards_smoothed_stats_three_step_tree_backup
	# np.save('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Project_652/Code/Linear_Approximator/WindyGridWorld_Results/'  + 'Three_Step_Tree_Backup' + '.npy', cum_rwd)
	# env.close()



	# print "On Policy Q(sigma)"
	# estimator = Estimator()
	# num_episodes = 1000
	# smoothing_window = 1
	# stats_q_sigma_on_policy = Q_Sigma_On_Policy(env, estimator, num_episodes, epsilon=0.1)
	# rewards_smoothed_stats_q_sigma_on_policy = pd.Series(stats_q_sigma_on_policy.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()	
	# cum_rwd = rewards_smoothed_stats_q_sigma_on_policy
	# np.save('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Project_652/Code/Linear_Approximator/WindyGridWorld_Results/'  + 'Q(sigma)_On_Policy' + '.npy', cum_rwd)
	# env.close()



	# print "Off Policy Q(sigma)"
	# theta = np.random.normal(size=(400,env.action_space.n))
	# num_episodes = 1000
	# smoothing_window = 1
	# stats_q_sigma_off_policy = Q_Sigma_Off_Policy(env, theta, num_episodes, epsilon=0.1)
	# rewards_smoothed_stats_q_sigma_off_policy = pd.Series(stats_q_sigma_off_policy.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()	
	# cum_rwd = rewards_smoothed_stats_q_sigma_off_policy
	# np.save('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Project_652/Code/Linear_Approximator/WindyGridWorld_Results/'  + 'Off_Policy_Q_Sigma' + '.npy', cum_rwd)
	# env.close()


	

	print "Off Policy Q(sigma) 2 Step"
	theta = np.random.normal(size=(400,env.action_space.n))
	num_episodes = 1000
	smoothing_window = 1
	stats_q_sigma_off_policy_2 = Q_Sigma_Off_Policy_2_Step(env, theta, num_episodes, epsilon=0.1)
	rewards_smoothed_stats_q_sigma_off_policy_2 = pd.Series(stats_q_sigma_off_policy_2.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()	
	cum_rwd = rewards_smoothed_stats_q_sigma_off_policy_2
	np.save('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Project_652/Code/Linear_Approximator/WindyGridWorld_Results/'  + 'Off_Policy_Q_Sigma_2_Step' + '.npy', cum_rwd)
	env.close()



	# print "Off Policy Q(sigma) 3 Step"
	# theta = np.random.normal(size=(400,env.action_space.n))
	# num_episodes = 1000
	# smoothing_window = 1
	# stats_q_sigma_off_policy_3 = Q_Sigma_Off_Policy_3_Step(env, theta, num_episodes, epsilon=0.1)
	# rewards_smoothed_stats_q_sigma_off_policy_3 = pd.Series(stats_q_sigma_off_policy_3.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()	
	# cum_rwd = rewards_smoothed_stats_q_sigma_off_policy_3
	# np.save('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Project_652/Code/Linear_Approximator/WindyGridWorld_Results/'  + 'Off_Policy_Q_Sigma_3_Step' + '.npy', cum_rwd)
	# env.close()


	
if __name__ == '__main__':
	main()




