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


from collections import defaultdict
from lib.envs.windy_gridworld import WindyGridworldEnv
from lib import plotting
env = WindyGridworldEnv()


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


def sarsa(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):

	Q = defaultdict(lambda: np.zeros(env.action_space.n))

	stats = plotting.EpisodeStats(episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes))
	action_space = env.action_space.n

	#on-policy which the agent follows - we want to optimize this policy function
	policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

	#each step in the episode
	for i_episode in range(num_episodes):

		print "Number of Episodes, SARSA", i_episode

		state = env.reset()
		action_probs = policy(state)

		#choose a from policy derived from Q (which is epsilon-greedy)
		action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

		#for every one step in the environment
		for t in itertools.count():
			#take a step in the environment
			# take action a, observe r and the next state
			next_state, reward, done, _ = env.step(action)

			#choose a' from s' using policy derived from Q
			next_action_probs = policy(next_state)
			next_action = np.random.choice(np.arange(len(action_probs)), p = next_action_probs)

			#update cumulative count of rewards based on action take (not next_action) using Q (epsilon-greedy)
			stats.episode_rewards[i_episode] += reward
			stats.episode_lengths[i_episode] = t

			# TD Update Equations
			#TD Target - One step ahead
			td_target = reward + discount_factor * Q[next_state][next_action]
			
			# TD Error
			td_delta = td_target - Q[state][action]

			Q[state][action] += alpha * td_delta
			
			if done:
				break
			
			action = next_action
			
			state = next_state


	return stats



def q_learning(env, num_episodes, discount_factor=1.0, alpha = 0.5, epsilon = 0.1):

	#Off Policy TD - Find Optimal Greedy policy while following epsilon-greedy policy

	Q = defaultdict(lambda : np.zeros(env.action_space.n))

	stats = plotting.EpisodeStats(episode_lengths=np.zeros(num_episodes),episode_rewards=np.zeros(num_episodes))  

	#policy that the agent is following
	policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)


	for i_episode in range(num_episodes):

		print "Number of Episodes, Q Learning", i_episode

		state = env.reset()

		for t in itertools.count():

			#take a step in the environmnet
			#choose action A using policy derived from Q
			action_probs = policy(state)
			action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
			
			# with taken aciton, observe the reward and the next state
			next_state, reward, done, _, = env.step(action)

			stats.episode_rewards[i_episode] += reward
			stats.episode_lengths[i_episode] = t


			# TD Update Equations:

			# max_a of Q(s', a) - where s' is the next state, and we consider all maximising over actions which was derived 
			#from previous policy based on Q
			best_next_action = np.argmax(Q[next_state])

			td_target = reward + discount_factor * Q[next_state][best_next_action]

			td_delta = td_target - Q[state][action]

			#update Q function based on the TD error
			Q[state][action] += alpha * td_delta

			if done:
				break

			state = next_state


	return stats




def double_q_learning(env, num_episodes, discount_factor=1.0, alpha = 0.5, epsilon = 0.1):

	#Off Policy TD - Find Optimal Greedy policy while following epsilon-greedy policy

	Q_A = defaultdict(lambda : np.zeros(env.action_space.n))

	Q_B = defaultdict(lambda : np.zeros(env.action_space.n))

	Total_Q = defaultdict(lambda : np.zeros(env.action_space.n))

	stats = plotting.EpisodeStats(episode_lengths=np.zeros(num_episodes),episode_rewards=np.zeros(num_episodes))  

	#choose a based on Q_A + Q_B
	policy = make_epsilon_greedy_policy(Total_Q, epsilon, env.action_space.n)


	for i_episode in range(num_episodes):

		print "Number of Episodes, Double Q Learning", i_episode

		state = env.reset()

		for t in itertools.count():

			#choose a from policy derived from Q1 + Q2 (epsilon greedy here)
			action_probs = policy(state)
			action = np.random.choice(np.arange(len(action_probs)), p=action_probs)			
			# with taken aciton, observe the reward and the next state
			next_state, reward, done, _, = env.step(action)

			stats.episode_rewards[i_episode] += reward
			stats.episode_lengths[i_episode] = t


			#choose randomly either update A or update B
			#randmly generate a for being 1 or 2
			random_number = random.randint(1,2)

			if random_number == 1:
				best_action_Q_A = np.argmax(Q_A[next_state])
				TD_Target_A = reward + discount_factor * Q_B[next_state][best_action_Q_A]
				TD_Delta_A = TD_Target_A - Q_A[state][action]
				Q_A[state][action] += alpha * TD_Delta_A

			elif random_number ==2:
				best_action_Q_B = np.argmax(Q_B[next_state])
				TD_Target_B = reward + discount_factor * Q_A[next_state][best_action_Q_B]
				TD_Delta_B = TD_Target_B - Q_B[state][action]
				Q_B[state][action] += alpha * TD_Delta_B


			if done:
				break

			state = next_state
			Total_Q[state][action] = Q_A[state][action] + Q_B[state][action]


	return stats




"""
Expected SARSA Algorithm = 1 step TREE BACKUP
"""

def one_step_tree_backup(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):

	#Expected SARSA : same algorithm steps as Q-Learning, 
	# only difference : instead of maximum over next state and action pairs
	# use the expected value
	Q = defaultdict(lambda : np.zeros(env.action_space.n))
	stats = plotting.EpisodeStats(episode_lengths=np.zeros(num_episodes),episode_rewards=np.zeros(num_episodes))  

	policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

	for i_episode in range(num_episodes):
		print "Number of Episodes, Expected SARSA", i_episode
		state = env.reset()


		#steps within each episode
		for t in itertools.count():
			#pick the first action
			#choose A from S using policy derived from Q (epsilon-greedy)
			action_probs = policy(state)
			action = np.random.choice(np.arange(len(action_probs)), p = action_probs)

			#reward and next state based on the action chosen according to epislon greedy policy
			next_state, reward, done, _ = env.step(action)
			
			#reward by taking action under the policy pi
			stats.episode_rewards[i_episode] += reward
			stats.episode_lengths[i_episode] = t


			#pick the next action
			# we want an expectation over the next actions 
			#take into account how likely each action is under the current policy

			next_action_probs = policy(next_state)
			next_action = np.random.choice(np.arange(len(next_action_probs)), p =next_action_probs )

			#V = sum_a pi(a, s_{t+1})Q(s_{t+1}, a)
			V = np.sum(next_action_probs * Q[next_state])

			#Update rule in Expected SARSA
			td_target = reward + discount_factor * V
			td_delta = td_target - Q[state][action]

			Q[state][action] += alpha * td_delta


			if done:
				break
			state = next_state

	return stats


"""
Two Step Tree Backup
"""

def two_step_tree_backup(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):


    Q = defaultdict(lambda : np.zeros(env.action_space.n))
    stats = plotting.EpisodeStats(episode_lengths=np.zeros(num_episodes),episode_rewards=np.zeros(num_episodes))  

    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    for i_episode in range(num_episodes):

    	print "Number of Episodes, Two Step Tree Backup", i_episode

        state = env.reset()

        #steps within each episode
        for t in itertools.count():
            #pick the first action
            #choose A from S using policy derived from Q (epsilon-greedy)
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p = action_probs)

            #reward and next state based on the action chosen according to epislon greedy policy
            next_state, reward, done , _ = env.step(action)
            
            #reward by taking action under the policy pi
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            next_action_probs = policy(next_state)
            next_action = np.random.choice(np.arange(len(next_action_probs)), p =next_action_probs )

            #V = sum_a pi(a, s_{t+1})Q(s_{t+1}, a)
            V = np.sum(next_action_probs * Q[next_state])


            next_next_state, next_reward, _, _ = env.step(next_action)
    
            next_next_action_probs = policy(next_next_state)
            next_next_action = np.random.choice(np.arange(len(next_next_action_probs)), p = next_next_action_probs)

            next_V = np.sum(next_next_action_probs * Q[next_next_state])            

            Delta = next_reward + discount_factor * next_V - Q[next_state][next_action]

            # print "Delta :", Delta

            # print "Next Action Prob ", np.max(next_action_probs)

            next_action_selection_probability = np.max(next_action_probs)

            td_target = reward + discount_factor * V +  discount_factor *  next_action_selection_probability * Delta

            td_delta = td_target - Q[state][action]

            Q[state][action] += alpha * td_delta


            if done:
                break

            state = next_state

    return stats


"""
Three Step Tree Backup
"""

def three_step_tree_backup(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):


	Q = defaultdict(lambda : np.zeros(env.action_space.n))
	stats = plotting.EpisodeStats(episode_lengths=np.zeros(num_episodes),episode_rewards=np.zeros(num_episodes))  

	for i_episode in range(num_episodes):

		print "Episode Number, Three Step Tree Backup:", i_episode
		#agent policy based on the greedy maximisation of Q
		policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
		last_reward = stats.episode_rewards[i_episode - 1]
		state = env.reset()



		#for each one step in the environment
		for t in itertools.count():

			action_probs = policy(state)
			action = np.random.choice(np.arange(len(action_probs)), p=action_probs)


			next_state, reward, done, _ = env.step(action)
			if done:
				break
			
			stats.episode_rewards[i_episode] += reward
			stats.episode_lengths[i_episode] = t

			next_action_probs = policy(next_state)
			next_action = np.random.choice(np.arange(len(next_action_probs)), p = next_action_probs)
			
			V = np.sum( next_action_probs * Q[next_state])

			Delta = reward + discount_factor * V - Q[state][action]


			next_next_state, next_reward, _, _ = env.step(next_action)
			next_next_action_probs = policy(next_next_state)
			next_next_action = np.random.choice(np.arange(len(next_next_action_probs)), p = next_next_action_probs)

			next_V = np.sum(next_next_action_probs * Q[next_next_state])		
			
			Delta_t_1 = next_reward + discount_factor * next_V - Q[next_state][next_action]

			next_next_next_state, next_next_reward, _, _ = env.step(next_next_action)
			next_next_next_action_probs = policy(next_next_next_state)
			next_next_next_action = np.random.choice(np.arange(len(next_next_next_action_probs)), p = next_next_next_action_probs)

			next_next_V = np.sum(next_next_next_action_probs * Q[next_next_next_state])

			Delta_t_2 = next_next_reward + discount_factor * next_next_V - Q[next_next_state][next_next_action]

			next_action_selection_probability = np.max(next_action_probs)
			next_next_action_selection_probability = np.max(next_next_action_probs)

			td_target = Q[state][action] + Delta + discount_factor * next_action_selection_probability * Delta_t_1 + discount_factor * discount_factor * next_action_selection_probability * next_next_action_selection_probability * Delta_t_2			
			td_delta = td_target - Q[state][action]

			Q[state][action] += alpha * td_delta			
			state = next_state

	return stats




"""
Q(sigma) algorithm
"""

def q_sigma_on_policy(env, num_episodes, discount_factor=1.0, alpha=0.1, epsilon=0.1):

    #Expected SARSA : same algorithm steps as Q-Learning, 
    # only difference : instead of maximum over next state and action pairs
    # use the expected value
    Q = defaultdict(lambda : np.zeros(env.action_space.n))
    stats = plotting.EpisodeStats(episode_lengths=np.zeros(num_episodes),episode_rewards=np.zeros(num_episodes))  

    # policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    for i_episode in range(num_episodes):

    	print "Number of Episodes, Q(sigma) On Policy", i_episode

    	policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
        state = env.reset()
        action_probs = policy(state)

        #choose a from policy derived from Q (which is epsilon-greedy)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)


        #steps within each episode
        for t in itertools.count():
            #take a step in the environment
            # take action a, observe r and the next state
            next_state, reward, done, _ = env.step(action)

            #reward by taking action under the policy pi
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t


            next_action_probs = policy(next_state)
            next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs )


            #define sigma to be a random variable between 0 and 1?
            sigma = random.randint(0,1)

            #V = sum_a pi(a, s_{t+1})Q(s_{t+1}, a)
            V = np.sum(next_action_probs * Q[next_state])

            Sigma_Effect = sigma * Q[next_state][next_action] + (1 - sigma) * V


            td_target = reward + discount_factor * Sigma_Effect

            td_delta = td_target - Q[state][action]

            Q[state][action] += alpha * td_delta

            if done:
                break
            action = next_action
            state = next_state

    return stats


def behaviour_policy_epsilon_greedy(Q, epsilon, nA):

	def policy_fn(observation):
		A = np.ones(nA, dtype=float) * epsilon/nA
		best_action = np.argmax(Q[observation])
		A[best_action] += ( 1.0 - epsilon)
		return A

	return policy_fn


# def behaviour_policy_epsilon_greedy(Q, tau, nA):

# 	def policy_fn(observation):
# 		exp_tau = Q[observation] / tau
# 		policy = np.exp(exp_tau) / np.sum(np.exp(exp_tau), axis=0)
# 		A = policy
# 		return A
# 	return policy_fn





from numpy.random import binomial
def binomial_sigma(p):
	sample = binomial(n=1, p=p)
	return sample



def Q_Sigma_Off_Policy(env, num_episodes, discount_factor=1.0, alpha=0.1, epsilon=0.1):

	Q = defaultdict(lambda : np.zeros(env.action_space.n))
	stats = plotting.EpisodeStats(episode_lengths=np.zeros(num_episodes),episode_rewards=np.zeros(num_episodes))  


	# policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
	tau = 1
	tau_decay = 0.999

	sigma = 1
	sigma_decay = 0.995

	for i_episode in range(num_episodes):

		print "Number of Episodes, Q(sigma) Off Policy", i_episode

		policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

		off_policy = behaviour_policy_epsilon_greedy(Q, tau, env.action_space.n)

		tau = tau * tau_decay

		if tau < 0.0001:
			tau = 0.0001

		state = env.reset()

		for t in itertools.count():

			action_probs = off_policy(state)

			action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

			state_t_1, reward, done, _ = env.step(action)

			if done:
				sigma = sigma * sigma_decay
				if sigma < 0.0001:
					sigma = 0.0001
				break				

			stats.episode_rewards[i_episode] += reward
			stats.episode_lengths[i_episode] = t

			# #select sigma value
			# probability = 0.5
			# sigma_t_1 = binomial_sigma(probability)

			sigma_t_1 = sigma

			#select next action based on the behaviour policy at next state
			next_action_probs = off_policy(state_t_1)
			action_t_1 = np.random.choice(np.arange(len(next_action_probs)), p = next_action_probs)


			on_policy_next_action_probs = policy(state_t_1)
			on_policy_a_t_1 = np.random.choice(np.arange(len(on_policy_next_action_probs)), p = on_policy_next_action_probs)
			V_t_1 = np.sum( on_policy_next_action_probs * Q[state_t_1] )

			Delta_t = reward + discount_factor * (  sigma_t_1 * Q[state_t_1][action_t_1] + (1 - sigma_t_1) * V_t_1  ) - Q[state][action]

			Q[state][action] += alpha * Delta_t


			state = state_t_1


	return stats



def Q_Sigma_Off_Policy_2_Step(env, num_episodes, discount_factor=1.0, alpha=0.1, epsilon=0.1):

	Q = defaultdict(lambda : np.zeros(env.action_space.n))
	stats = plotting.EpisodeStats(
		episode_lengths=np.zeros(num_episodes),
		episode_rewards=np.zeros(num_episodes))  


	tau = 1
	tau_decay = 0.999

	sigma = 1
	sigma_decay = 0.995

	for i_episode in range(num_episodes):

		print "Number of Episodes, Q(sigma) Off Policy 2 Step", i_episode

		policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
		off_policy = behaviour_policy_epsilon_greedy(Q, tau, env.action_space.n)

		tau = tau * tau_decay

		if tau < 0.0001:
			tau = 0.0001

		state = env.reset()

		for t in itertools.count():
			action_probs = off_policy(state)
			action = np.random.choice(np.arange(len(action_probs)), p = action_probs)

			state_t_1, reward, done, _ = env.step(action)

			stats.episode_rewards[i_episode] += reward
			stats.episode_lengths[i_episode] = t

			if done:
				sigma = sigma * sigma_decay
				if sigma < 0.0001:
					sigma = 0.0001
				break		

			# probability = 0.5
			# sigma_t_1 = binomial_sigma(probability)

			sigma_t_1 = sigma

			next_action_probs = off_policy(state_t_1)
			action_t_1 = np.random.choice(np.arange(len(next_action_probs)), p = next_action_probs)

			on_policy_next_action_probs = policy(state_t_1)
			on_policy_a_t_1 = np.random.choice(np.arange(len(on_policy_next_action_probs)), p = on_policy_next_action_probs)
			V_t_1 = np.sum( on_policy_next_action_probs * Q[state_t_1] )

			Delta_t = reward + discount_factor * (  sigma_t_1 * Q[state_t_1][action_t_1] + (1 - sigma_t_1) * V_t_1  ) - Q[state][action]


			state_t_2, next_reward, _, _ = env.step(action_t_1)

			next_next_action_probs = off_policy(state_t_2)
			action_t_2 = np.random.choice(np.arange(len(next_next_action_probs)), p = next_next_action_probs)

			on_policy_next_next_action_probs = policy(state_t_2)
			on_policy_a_t_2 = np.random.choice(np.arange(len(on_policy_next_next_action_probs)), p = on_policy_next_next_action_probs)
			V_t_2 = np.sum( on_policy_next_next_action_probs * Q[state_t_2])

			sigma_t_2 = sigma

			Delta_t_1 = next_reward + discount_factor * (  sigma_t_2 * Q[state_t_2][action_t_2] + (1 - sigma_t_2) * V_t_2  ) - Q[state_t_1][action_t_1]


			"""
			2 step TD Target --- G_t(2)
			"""
			
			on_policy_action_probability = on_policy_next_action_probs[on_policy_a_t_1]
			off_policy_action_probability = next_action_probs[action_t_1]

			td_target = Q[state][action] + Delta_t + discount_factor * ( (1 - sigma_t_1) *  on_policy_action_probability + sigma_t_1 ) * Delta_t_1

			"""
			Computing Importance Sampling Ratio
			"""
			rho = np.divide( on_policy_action_probability, off_policy_action_probability )
			rho_sigma = sigma_t_1 * rho + 1 - sigma_t_1

			td_error = td_target - Q[state][action]

			Q[state][action] += alpha * rho_sigma * td_error

			state = state_t_1

	return stats




def Q_Sigma_Off_Policy_3_Step(env, num_episodes, discount_factor=1.0, alpha=0.1, epsilon=0.1):

	Q = defaultdict(lambda : np.zeros(env.action_space.n))

	stats = plotting.EpisodeStats(
		episode_lengths=np.zeros(num_episodes),
		episode_rewards=np.zeros(num_episodes))  

	tau = 1
	tau_decay = 0.999

	sigma = 1
	sigma_decay = 0.995

	for i_episode in range(num_episodes):

		print "Number of Episodes, Q(sigma) Off Policy 3 Step", i_episode

		policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
		off_policy = behaviour_policy_epsilon_greedy(Q, tau, env.action_space.n)

		tau = tau * tau_decay

		if tau < 0.0001:
			tau = 0.0001

		state = env.reset()

		for t in itertools.count():
			action_probs = off_policy(state)
			action = np.random.choice(np.arange(len(action_probs)), p = action_probs)

			state_t_1, reward, done, _ = env.step(action)

			stats.episode_rewards[i_episode] += reward
			stats.episode_lengths[i_episode] = t

			if done:
				sigma = sigma * sigma_decay
				if sigma < 0.0001:
					sigma = 0.0001				
				break		

			# probability = 0.5
			# sigma_t_1 = binomial_sigma(probability)

			sigma_t_1 = sigma

			next_action_probs = off_policy(state_t_1)
			action_t_1 = np.random.choice(np.arange(len(next_action_probs)), p = next_action_probs)

			on_policy_next_action_probs = policy(state_t_1)
			on_policy_a_t_1 = np.random.choice(np.arange(len(on_policy_next_action_probs)), p = on_policy_next_action_probs)
			V_t_1 = np.sum( on_policy_next_action_probs * Q[state_t_1] )

			Delta_t = reward + discount_factor * (  sigma_t_1 * Q[state_t_1][action_t_1] + (1 - sigma_t_1) * V_t_1  ) - Q[state][action]


			state_t_2, next_reward, _, _ = env.step(action_t_1)

			next_next_action_probs = off_policy(state_t_2)
			action_t_2 = np.random.choice(np.arange(len(next_next_action_probs)), p = next_next_action_probs)

			on_policy_next_next_action_probs = policy(state_t_2)
			on_policy_a_t_2 = np.random.choice(np.arange(len(on_policy_next_next_action_probs)), p = on_policy_next_next_action_probs)
			V_t_2 = np.sum( on_policy_next_next_action_probs * Q[state_t_2])

			sigma_t_2 = sigma

			Delta_t_1 = next_reward + discount_factor * (  sigma_t_2 * Q[state_t_2][action_t_2] + (1 - sigma_t_2) * V_t_2  ) - Q[state_t_1][action_t_1]

			"""
			3 step TD Target --- G_t(2)
			"""
			state_t_3, next_next_reward, _, _ = env.step(action_t_2)

			next_next_next_action_probs = off_policy(state_t_3)
			action_t_3 = np.random.choice(np.arange(len(next_next_next_action_probs)), p = next_next_next_action_probs)


			on_policy_next_next_next_action_probs = policy(state_t_3)
			on_policy_a_t_3 = np.random.choice(np.arange(len(on_policy_next_next_next_action_probs)), p = on_policy_next_next_next_action_probs)
			V_t_3 = np.sum(on_policy_next_next_next_action_probs * Q[state_t_3])

			sigma_t_3 = sigma

			Delta_t_2 = next_next_reward + discount_factor * (sigma_t_3 * Q[state_t_3][action_t_3] + (1 - sigma_t_3) * V_t_3 ) -  Q[state_t_2][action_t_2]


			on_policy_action_probability = on_policy_next_action_probs[on_policy_a_t_1]
			off_policy_action_probability = next_action_probs[action_t_1]

			on_policy_next_action_probability = on_policy_next_next_action_probs[on_policy_a_t_2]
			off_policy_next_action_probability = next_next_action_probs[action_t_2]



			td_target = Q[state][action] + Delta_t + discount_factor * ( (1 - sigma_t_1) *  on_policy_action_probability + sigma_t_1 ) * Delta_t_1 + discount_factor * ( (1 - sigma_t_2)  * on_policy_next_action_probability + sigma_t_2 ) * Delta_t_2

			"""
			Computing Importance Sampling Ratio
			"""
			rho = np.divide( on_policy_action_probability, off_policy_action_probability )
			rho_1 = np.divide( on_policy_next_action_probability, off_policy_next_action_probability )

			rho_sigma = sigma_t_1 * rho + 1 - sigma_t_1
			rho_sigma_1 = sigma_t_2 * rho_1 + 1 - sigma_t_2

			all_rho_sigma = rho_sigma * rho_sigma_1

			td_error = td_target -  Q[state][action]

			Q[state][action] += alpha * all_rho_sigma * td_error 

			state = state_t_1

	return stats			



"""
Code for algorithms with Eligiblity Traces (Backward View)
"""

def sarsa_lambda(env, num_episodes, discount_factor=1.0, alpha=0.1, epsilon=0.1, lambda_param = 0.1):

	Q = defaultdict(lambda : np.zeros(env.action_space.n))

	stats = plotting.EpisodeStats(
		episode_lengths=np.zeros(num_episodes),
		episode_rewards=np.zeros(num_episodes))  

	for i_episode in range(num_episodes):

		print "Number of Episodes, SARSA(lambda)", i_episode

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

			Delta = reward + discount_factor * Q[next_state][next_action] - Q[state][action]

			eligibility[state][action] = eligibility[state][action] + 1


			for s in range(env.observation_space.n):
				for a in range(env.action_space.n):
					Q[s][a] = Q[s][a] + alpha * Delta * eligibility[s][a]
					eligibility[s][a] = discount_factor * lambda_param * eligibility[s][a]

			if done:
				break

			action = next_action
			state = next_state

	return stats



def q_lambda_watkins(env, num_episodes, discount_factor=1.0, alpha=0.1, epsilon=0.1, lambda_param = 0.1):

	Q = defaultdict(lambda : np.zeros(env.action_space.n))

	stats = plotting.EpisodeStats(
		episode_lengths=np.zeros(num_episodes),
		episode_rewards=np.zeros(num_episodes))  

	for i_episode in range(num_episodes):

		print "Number of Episodes, Watkins Q(lambda)", i_episode

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

			best_action = np.argmax(Q[next_state])

			Delta = reward + discount_factor * Q[next_state][best_action] - Q[state][action]

			eligibility[state][action] = eligibility[state][action] + 1


			for s in range(env.observation_space.n):
				for a in range(env.action_space.n):
					Q[s][a] = Q[s][a] + alpha * Delta * eligibility[s][a]

					if next_action == best_action:
						eligibility[s][a] = discount_factor * lambda_param * eligibility[s][a]
					else:
						eligibility[s][a] = 0

			if done:
				break

			action = next_action
			state = next_state

	return stats




def q_lambda_naive(env, num_episodes, discount_factor=1.0, alpha=0.1, epsilon=0.1, lambda_param = 0.1):

	Q = defaultdict(lambda : np.zeros(env.action_space.n))

	stats = plotting.EpisodeStats(
		episode_lengths=np.zeros(num_episodes),
		episode_rewards=np.zeros(num_episodes))  

	for i_episode in range(num_episodes):

		print "Number of Episodes, Naive Q(lambda)", i_episode

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


			best_action = np.argmax(Q[next_state])

			Delta = reward + discount_factor * Q[next_state][best_action] - Q[state][action]

			eligibility[state][action] = eligibility[state][action] + 1


			for s in range(env.observation_space.n):
				for a in range(env.action_space.n):
					Q[s][a] = Q[s][a] + alpha * Delta * eligibility[s][a]
					eligibility[s][a] = discount_factor * lambda_param * eligibility[s][a]


			if done:
				break

			action = next_action
			state = next_state

	return stats









"""
Online Tree Backup with Eligiblity Traces
"""
def tree_backup_lambda(env, num_episodes, discount_factor=1.0, alpha=0.1, epsilon=0.1, lambda_param = 0.1):

	Q = defaultdict(lambda : np.zeros(env.action_space.n))

	stats = plotting.EpisodeStats(
		episode_lengths=np.zeros(num_episodes),
		episode_rewards=np.zeros(num_episodes))  

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

			eligibility[state][action] = 1

			# Q[state][action] = Q[state][action] + alpha * Delta * eligibility[state][action]


			for s in range(env.observation_space.n):
				for a in range(env.action_space.n):
					Q[s][a] = Q[s][a] + alpha * Delta * eligibility[s][a]
					eligibility[s][a] = eligibility[s][a] * discount_factor * lambda_param * action_probability


			if done:
				break

			action = next_action
			state = next_state

	return stats



"""
Q(sigma) algorithm with eligibility traces

NO PROOF OR ALGO YET - DISCUSS WITH DOINA AND RICH

"""

def q_sigma_lambda(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1, lambda_param=0.1):
	
	Q = defaultdict(lambda : np.zeros(env.action_space.n))

	stats = plotting.EpisodeStats(
		episode_lengths=np.zeros(num_episodes),
		episode_rewards=np.zeros(num_episodes))  


	for i_episode in range(num_episodes):

		print "Number of Episodes, Q(sigma)(lambda) with Eligiblity Traces", i_episode

		policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
		state = env.reset()
		
		action_probs = policy(state)
		action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

		eligibility = defaultdict(lambda : np.zeros(env.action_space.n))	


		for t in itertools.count():

			next_state, reward, done, _ = env.step(action)

			stats.episode_rewards[i_episode] += reward
			stats.episode_lengths[i_episode] = t

	        next_action_probs = policy(next_state)
	        next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs )
	        action_probability = next_action_probs[next_action]

	        probability = 0.5
	        sigma = binomial_sigma(probability)       

	        V = np.sum(next_action_probs * Q[next_state])

	        td_target = reward + discount_factor * (sigma * Q[next_state][next_action] + (1 - sigma) * V)

	        Delta = td_target - Q[state][action]

	        eligibility[state][action] = eligibility[state][action] + 1

	        # Q[state][action] = Q[state][action] + alpha * Delta * eligibility[state][action]


	        for s in range(env.observation_space.n):
	        	for a in range(env.action_space.n):
	        		Q[s][a] = Q[s][a] + alpha * Delta * eligibility[s][a]
	        		eligibility[s][a] = eligibility[s][a] * discount_factor * lambda_param * action_probability


	        if done:
	        	break

	        action = next_action
	        state = next_state

	return stats			







"""
TRUE ONLINE TD(LAMBDA)
"""


def true_online_sarsa_lambda(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1, lambda_param=0.1):
	
	Q = defaultdict(lambda : np.zeros(env.action_space.n))

	stats = plotting.EpisodeStats(
		episode_lengths=np.zeros(num_episodes),
		episode_rewards=np.zeros(num_episodes))  


	for i_episode in range(num_episodes):

		print "Number of Episodes, Q(sigma)(lambda) with Eligiblity Traces", i_episode

		policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
		state = env.reset()
		
		action_probs = policy(state)
		action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

		eligibility = defaultdict(lambda : np.zeros(env.action_space.n))	


		for t in itertools.count():

			next_state, reward, done, _ = env.step(action)

			stats.episode_rewards[i_episode] += reward
			stats.episode_lengths[i_episode] = t

	        next_action_probs = policy(next_state)
	        next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs )
	        action_probability = next_action_probs[next_action]

	        probability = 0.5
	        sigma = binomial_sigma(probability)       

	        V = np.sum(next_action_probs * Q[next_state])

	        td_target = reward + discount_factor * (sigma * Q[next_state][next_action] + (1 - sigma) * V)

	        Delta = td_target - Q[state][action]

	        eligibility[state][action] = eligibility[state][action] + 1

	        # Q[state][action] = Q[state][action] + alpha * Delta * eligibility[state][action]


	        for s in range(env.observation_space.n):
	        	for a in range(env.action_space.n):
	        		Q[s][a] = Q[s][a] + alpha * Delta * eligibility[s][a]
	        		eligibility[s][a] = eligibility[s][a] * discount_factor * lambda_param * action_probability


	        if done:
	        	break

	        action = next_action
	        state = next_state

	return stats			



from collections import defaultdict
from lib.envs.windy_gridworld import WindyGridworldEnv
from lib import plotting
env = WindyGridworldEnv()


def main():

	# print "SARSA"
	# env = WindyGridworldEnv()
	# num_episodes = 2000
	# smoothing_window = 1
	# stats_sarsa = sarsa(env, num_episodes)
	# rewards_sarsa = pd.Series(stats_sarsa.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
	# cum_rwd = rewards_sarsa
	# np.save('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Project_652/Code/Tabular/WindyGridWorld_Results/'  + 'SARSA' + '.npy', cum_rwd)
	# # plotting.plot_episode_stats(stats_sarsa)
	# env.close()


	# print "Q Learning"
	# env = WindyGridworldEnv()
	# num_episodes = 2000
	# smoothing_window = 1
	# stats_q_learning = q_learning(env, num_episodes)
	# rewards_q_learning = pd.Series(stats_q_learning.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
	# cum_rwd = rewards_q_learning
	# np.save('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Project_652/Code/Tabular/WindyGridWorld_Results/'  + 'Q_Learning' + '.npy', cum_rwd)
	# # plotting.plot_episode_stats(stats_q_learning)
	# env.close()

	# print "Double Q Learning"
	# env = WindyGridworldEnv()
	# num_episodes = 2000
	# smoothing_window = 1
	# stats_double_q_learning = double_q_learning(env, num_episodes)
	# rewards_double_q_learning = pd.Series(stats_double_q_learning.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
	# cum_rwd = rewards_double_q_learning
	# np.save('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Project_652/Code/Tabular/WindyGridWorld_Results/'  + 'Double_Q_Learning' + '.npy', cum_rwd)
	# # plotting.plot_episode_stats(stats_double_q_learning)
	# env.close()


	# print "One Step Tree Backup (Expected SARSA)"
	# env = WindyGridworldEnv()
	# num_episodes = 2000
	# smoothing_window = 1
	# stats_expected_sarsa = one_step_tree_backup(env, num_episodes)
	# rewards_expected_sarsa = pd.Series(stats_expected_sarsa.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
	# cum_rwd = rewards_expected_sarsa
	# np.save('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Project_652/Code/Tabular/WindyGridWorld_Results/'  + 'One_Step_Tree_Backup' + '.npy', cum_rwd)
	# # plotting.plot_episode_stats(stats_expected_sarsa)
	# env.close()

	# print "Two Step Tree Backup"
	# env = WindyGridworldEnv()
	# num_episodes = 2000
	# smoothing_window = 1
	# stats_two_step_tree_backup = two_step_tree_backup(env, num_episodes)
	# rewards_two_step_tree_backup = pd.Series(stats_two_step_tree_backup.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
	# cum_rwd = rewards_two_step_tree_backup
	# np.save('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Project_652/Code/Tabular/WindyGridWorld_Results/'  + 'Two_Step_Tree_Backup' + '.npy', cum_rwd)
	# # plotting.plot_episode_stats(stats_two_step_tree_backup)
	# env.close()

	# print "Three Step Tree Backup"
	# env = WindyGridworldEnv()
	# num_episodes = 2000
	# smoothing_window = 1
	# stats_three_step_tree_backup = three_step_tree_backup(env, num_episodes)
	# rewards_three_step_tree_backup = pd.Series(stats_three_step_tree_backup.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
	# cum_rwd = rewards_three_step_tree_backup
	# np.save('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Project_652/Code/Tabular/WindyGridWorld_Results/'  + 'Three_Step_Tree_Backup' + '.npy', cum_rwd)
	# # plotting.plot_episode_stats(stats_three_step_tree_backup)
	# env.close()


	# print "Q(sigma) On Policy"
	# env = WindyGridworldEnv()
	# num_episodes = 2000
	# smoothing_window = 1
	# stats_q_sigma_on_policy = q_sigma_on_policy(env, num_episodes)
	# rewards_stats_q_sigma_on_policy = pd.Series(stats_q_sigma_on_policy.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
	# cum_rwd = rewards_stats_q_sigma_on_policy
	# np.save('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Project_652/Code/Tabular/WindyGridWorld_Results/'  + 'Q_Sigma_On_Policy' + '.npy', cum_rwd)
	# # plotting.plot_episode_stats(stats_q_sigma_on_policy)
	# env.close()


	# print "Q(sigma) Off Policy"
	# env = WindyGridworldEnv()
	# num_episodes = 2000
	# smoothing_window = 1
	# stats_q_sigma_off_policy = Q_Sigma_Off_Policy(env, num_episodes)
	# rewards_stats_q_sigma_off_policy = pd.Series(stats_q_sigma_off_policy.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
	# cum_rwd = rewards_stats_q_sigma_off_policy
	# np.save('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Project_652/Code/Tabular/WindyGridWorld_Results/'  + 'Q_Sigma_Off_Policy' + '.npy', cum_rwd)
	# # plotting.plot_episode_stats(stats_q_sigma_off_policy)
	# env.close()


	# print "Q(sigma) Off Policy 2 Step"
	# env = WindyGridworldEnv()
	# num_episodes = 2000
	# smoothing_window = 1
	# stats_q_sigma_off_policy_2_step = Q_Sigma_Off_Policy_2_Step(env, num_episodes)
	# rewards_stats_q_sigma_off_policy_2 = pd.Series(stats_q_sigma_off_policy_2_step.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
	# cum_rwd = rewards_stats_q_sigma_off_policy_2
	# np.save('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Project_652/Code/Tabular/WindyGridWorld_Results/'  + 'Q_Sigma_Off_Policy_2_Step' + '.npy', cum_rwd)
	# # plotting.plot_episode_stats(stats_q_sigma_off_policy_2_step)
	# env.close()


	# print "Q(sigma) Off Policy 3 Step"
	# env = WindyGridworldEnv()
	# num_episodes = 2000
	# smoothing_window = 1
	# stats_q_sigma_off_policy_3_step = Q_Sigma_Off_Policy_3_Step(env, num_episodes)
	# rewards_stats_q_sigma_off_policy_3 = pd.Series(stats_q_sigma_off_policy_3_step.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
	# cum_rwd = rewards_stats_q_sigma_off_policy_3
	# np.save('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Project_652/Code/Tabular/WindyGridWorld_Results/'  + 'Q_Sigma_Off_Policy_3_Step' + '.npy', cum_rwd)
	# # plotting.plot_episode_stats(stats_q_sigma_off_policy_3_step)
	# env.close()


	# print "SARSA(lambda)"
	# env = WindyGridworldEnv()
	# num_episodes = 2000
	# smoothing_window = 1
	# stats_sarsa_lambda = sarsa_lambda(env, num_episodes)
	# rewards_stats_sarsa_lambda = pd.Series(stats_sarsa_lambda.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
	# cum_rwd = rewards_stats_sarsa_lambda
	# np.save('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Project_652/Code/Tabular/WindyGridWorld_Results/'  + 'Sarsa(lambda)' + '.npy', cum_rwd)
	# # plotting.plot_episode_stats(stats_sarsa_lambda)
	# env.close()



	# print "Watkins Q(lambda)"
	# env = WindyGridworldEnv()
	# num_episodes = 2000
	# smoothing_window = 1
	# stats_q_lambda = q_lambda_watkins(env, num_episodes)
	# rewards_stats_q_lambda = pd.Series(stats_q_lambda.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
	# cum_rwd = rewards_stats_q_lambda
	# np.save('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Project_652/Code/Tabular/WindyGridWorld_Results/'  + 'Watkins Q(lambda)' + '.npy', cum_rwd)
	# # plotting.plot_episode_stats(stats_q_lambda)
	# env.close()


	# print "Naive Q(lambda)"
	# env = WindyGridworldEnv()
	# num_episodes = 2000
	# smoothing_window = 1
	# stats_q_lambda_naive = q_lambda_naive(env, num_episodes)
	# rewards_stats_q_naive = pd.Series(stats_q_lambda_naive.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
	# cum_rwd = rewards_stats_q_naive
	# np.save('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Project_652/Code/Tabular/WindyGridWorld_Results/'  + 'Naive Q(lambda)' + '.npy', cum_rwd)
	# # plotting.plot_episode_stats(stats_q_lambda_naive)
	# env.close()


	print "Tree Backup(lambda)"
	env = WindyGridworldEnv()
	num_episodes = 2000
	smoothing_window = 1
	stats_tree_lambda = tree_backup_lambda(env, num_episodes)
	rewards_stats_tree_lambda = pd.Series(stats_tree_lambda.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
	cum_rwd = rewards_stats_tree_lambda
	np.save('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Project_652/Code/Tabular/WindyGridWorld_Results/'  + 'Tree Backup(lambda)' + '.npy', cum_rwd)
	# plotting.plot_episode_stats(stats_tree_lambda)
	env.close()


	# print "Q(sigma)(lambda)"
	# num_episodes = 2000
	# smoothing_window = 1
	# stats_q_sigma_lambda = q_sigma_lambda(env, num_episodes)
	# rewards_stats_q_sigma_lambda = pd.Series(stats_q_sigma_lambda.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
	# cum_rwd = rewards_stats_q_sigma_lambda
	# np.save('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Project_652/Code/Tabular/WindyGridWorld_Results/'  + 'Q(sigma_lambda)' + '.npy', cum_rwd)
	# plotting.plot_episode_stats(stats_q_sigma_lambda)
	# env.close()





if __name__ == '__main__':
	main()









