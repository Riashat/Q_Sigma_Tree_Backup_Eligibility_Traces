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


def behaviour_policy_epsilon_greedy(Q, epsilon, nA):

	def policy_fn(observation):
		A = np.ones(nA, dtype=float) * epsilon/nA
		best_action = np.argmax(Q[observation])
		A[best_action] += ( 1.0 - epsilon)
		return A

	return policy_fn


def behaviour_policy_Boltzmann(Q, tau, nA):

	def policy_fn(observation):
		exp_tau = Q[observation] / tau
		policy = np.exp(exp_tau) / np.sum(np.exp(exp_tau), axis=0)
		A = policy
		return A
	return policy_fn



from numpy.random import binomial
def binomial_sigma(p):
	sample = binomial(n=1, p=p)
	return sample




"""
Q(sigma) algorithm
"""

def q_sigma_on_policy_eligibility_traces_static_sigma(env, num_episodes, discount_factor=1.0, alpha=0.1, epsilon=0.1, lambda_param = 0.1):

    #Expected SARSA : same algorithm steps as Q-Learning, 
    # only difference : instead of maximum over next state and action pairs
    # use the expected value
    Q = defaultdict(lambda : np.zeros(env.action_space.n))
    stats = plotting.EpisodeStats(episode_lengths=np.zeros(num_episodes),episode_rewards=np.zeros(num_episodes))  


    # policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    for i_episode in range(num_episodes):

    	print "Number of Episodes, Q(sigma) On Policy Eligibility Traces, Static Sigma", i_episode

    	policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
        state = env.reset()
        action_probs = policy(state)

        #choose a from policy derived from Q (which is epsilon-greedy)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)


        #initialise eligiblity traces
        eligibility = defaultdict(lambda : np.zeros(env.action_space.n))	

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
            action_probability = next_action_probs[next_action]

            probability = 0.5
            sigma = binomial_sigma(probability)

            # #define sigma to be a random variable between 0 and 1?
            # sigma = random.randint(0,1)

            #V = sum_a pi(a, s_{t+1})Q(s_{t+1}, a)
            V = np.sum(next_action_probs * Q[next_state])

            Sigma_Effect = sigma * Q[next_state][next_action] + (1 - sigma) * V


            td_target = reward + discount_factor * Sigma_Effect

            td_delta = td_target - Q[state][action]

            eligibility[state][action] = 1


            for s in range(env.observation_space.n):
            	for a in range(env.action_space.n):
            		Q[s][a] = Q[s][a] + alpha * td_delta * eligibility[s][a]
            		eligibility[s][a] = ( 1 - sigma) * eligibility[s][a] * discount_factor * lambda_param * action_probability + sigma *  discount_factor * lambda_param * eligibility[s][a]


            # Q[state][action] += alpha * td_delta

            if done:
                break
            action = next_action
            state = next_state

    return stats



def q_sigma_on_policy_eligibility_traces_dynamic_sigma(env, num_episodes, discount_factor=1.0, alpha=0.1, epsilon=0.1, lambda_param = 0.1):

    #Expected SARSA : same algorithm steps as Q-Learning, 
    # only difference : instead of maximum over next state and action pairs
    # use the expected value
    Q = defaultdict(lambda : np.zeros(env.action_space.n))
    stats = plotting.EpisodeStats(episode_lengths=np.zeros(num_episodes),episode_rewards=np.zeros(num_episodes))  

    # tau = 1
    # tau_decay = 0.999

    sigma = 1
    sigma_decay = 0.995


    # policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    for i_episode in range(num_episodes):

    	print "Number of Episodes, Q(sigma) On Policy Eligibility Traces, Dynamic Sigma", i_episode

    	policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
        state = env.reset()
        action_probs = policy(state)

        #choose a from policy derived from Q (which is epsilon-greedy)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)


        #initialise eligiblity traces
        eligibility = defaultdict(lambda : np.zeros(env.action_space.n))	

        #steps within each episode
        for t in itertools.count():
            #take a step in the environment
            # take action a, observe r and the next state
            next_state, reward, done, _ = env.step(action)
            if done:
            	sigma = sigma * sigma_decay

            	if sigma < 0.0001:
            		sigma = 0.0001

            	break

            #reward by taking action under the policy pi
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t


            next_action_probs = policy(next_state)
            next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs )
            action_probability = next_action_probs[next_action]

            #V = sum_a pi(a, s_{t+1})Q(s_{t+1}, a)
            V = np.sum(next_action_probs * Q[next_state])

            Sigma_Effect = sigma * Q[next_state][next_action] + (1 - sigma) * V


            td_target = reward + discount_factor * Sigma_Effect

            td_delta = td_target - Q[state][action]

            eligibility[state][action] = 1


            for s in range(env.observation_space.n):
            	for a in range(env.action_space.n):
            		Q[s][a] = Q[s][a] + alpha * td_delta * eligibility[s][a]
            		eligibility[s][a] = ( 1 - sigma) * eligibility[s][a] * discount_factor * lambda_param * action_probability + sigma *  discount_factor * lambda_param * eligibility[s][a]


            action = next_action
            state = next_state

    return stats



def q_sigma_off_policy_eligibility_traces_static_sigma(env, num_episodes, discount_factor=1.0, alpha=0.1, epsilon=0.1, lambda_param = 0.1):

    #Expected SARSA : same algorithm steps as Q-Learning, 
    # only difference : instead of maximum over next state and action pairs
    # use the expected value
    Q = defaultdict(lambda : np.zeros(env.action_space.n))
    stats = plotting.EpisodeStats(episode_lengths=np.zeros(num_episodes),episode_rewards=np.zeros(num_episodes))  

    tau = 1
    tau_decay = 0.999

    for i_episode in range(num_episodes):

    	print "Number of Episodes, Q(sigma) Off Policy Eligibility Traces, Static Sigma", i_episode

    	policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    	off_policy = behaviour_policy_Boltzmann(Q, tau, env.action_space.n)

    	tau = tau * tau_decay

    	if tau < 0.0001:
    		tau = 0.0001

        state = env.reset()
        action_probs = policy(state)

        #choose a from policy derived from Q (which is epsilon-greedy)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)


        #initialise eligiblity traces
        eligibility = defaultdict(lambda : np.zeros(env.action_space.n))	

        #steps within each episode
        for t in itertools.count():
            #take a step in the environment
            # take action a, observe r and the next state
            next_state, reward, done, _ = env.step(action)

            #reward by taking action under the policy pi
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            next_action_probs = off_policy(next_state)
            next_action = np.random.choice(np.arange(len(next_action_probs)), p = next_action_probs)
            action_probability = next_action_probs[next_action]


            on_policy_next_action_probs = policy(next_state)
            on_policy_next_action = np.random.choice(np.arange(len(on_policy_next_action_probs)), p = on_policy_next_action_probs)
            V_t_1 = np.sum( on_policy_next_action_probs * Q[next_state] )


            probability = 0.5
            sigma = binomial_sigma(probability)

            Delta_t = reward + discount_factor * (  sigma * Q[next_state][next_action] + (1 - sigma) * V_t_1  ) - Q[state][action]


            eligibility[state][action] = 1


            for s in range(env.observation_space.n):
            	for a in range(env.action_space.n):
            		Q[s][a] = Q[s][a] + alpha * Delta_t * eligibility[s][a]
            		eligibility[s][a] = ( 1 - sigma) * eligibility[s][a] * discount_factor * lambda_param * action_probability + sigma *  discount_factor * lambda_param * eligibility[s][a]


            if done:
                break

            action = next_action
            state = next_state

    return stats





def q_sigma_off_policy_eligibility_traces_dynamic_sigma(env, num_episodes, discount_factor=1.0, alpha=0.1, epsilon=0.1, lambda_param = 0.1):

    #Expected SARSA : same algorithm steps as Q-Learning, 
    # only difference : instead of maximum over next state and action pairs
    # use the expected value
    Q = defaultdict(lambda : np.zeros(env.action_space.n))
    stats = plotting.EpisodeStats(episode_lengths=np.zeros(num_episodes),episode_rewards=np.zeros(num_episodes))  

    tau = 1
    tau_decay = 0.999

    sigma = 1
    sigma_decay = 0.995


    for i_episode in range(num_episodes):

    	print "Number of Episodes, Q(sigma) Off Policy Eligibility Traces, Dynamic Sigma", i_episode

    	policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    	off_policy = behaviour_policy_Boltzmann(Q, tau, env.action_space.n)

    	tau = tau * tau_decay

    	if tau < 0.0001:
    		tau = 0.0001

        state = env.reset()
        action_probs = policy(state)

        #choose a from policy derived from Q (which is epsilon-greedy)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)


        #initialise eligiblity traces
        eligibility = defaultdict(lambda : np.zeros(env.action_space.n))	

        #steps within each episode
        for t in itertools.count():
            #take a step in the environment
            # take action a, observe r and the next state
            next_state, reward, done, _ = env.step(action)

            if done:
            	sigma = sigma * sigma_decay

            	if sigma < 0.0001:
            		sigma = 0.0001

            	break

            #reward by taking action under the policy pi
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            next_action_probs = off_policy(next_state)
            next_action = np.random.choice(np.arange(len(next_action_probs)), p = next_action_probs)
            action_probability = next_action_probs[next_action]


            on_policy_next_action_probs = policy(next_state)
            on_policy_next_action = np.random.choice(np.arange(len(on_policy_next_action_probs)), p = on_policy_next_action_probs)
            V_t_1 = np.sum( on_policy_next_action_probs * Q[next_state] )


            # probability = 0.5
            # sigma = binomial_sigma(probability)

            Delta_t = reward + discount_factor * (  sigma * Q[next_state][next_action] + (1 - sigma) * V_t_1  ) - Q[state][action]

            eligibility[state][action] = 1


            for s in range(env.observation_space.n):
            	for a in range(env.action_space.n):
            		Q[s][a] = Q[s][a] + alpha * Delta_t * eligibility[s][a]
            		eligibility[s][a] = ( 1 - sigma) * eligibility[s][a] * discount_factor * lambda_param * action_probability + sigma *  discount_factor * lambda_param * eligibility[s][a]




            action = next_action
            state = next_state

    return stats






def main():


	print "Q(sigma) On Policy Eligiblity Traces, Static Sigma"
	env = WindyGridworldEnv()
	num_episodes = 2000
	smoothing_window = 1
	stats_q_sigma_on_policy = q_sigma_on_policy_eligibility_traces_static_sigma(env, num_episodes)
	rewards_stats_q_sigma_on_policy = pd.Series(stats_q_sigma_on_policy.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
	cum_rwd = rewards_stats_q_sigma_on_policy
	np.save('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Project_652/Code/Tabular/WindyGridWorld_Results/'  + 'Q_Sigma_On_Policy_Eligibility_Traces_Static_Sigma' + '.npy', cum_rwd)
	# plotting.plot_episode_stats(stats_q_sigma_on_policy)
	env.close()

	print "Q(sigma) On Policy Eligiblity Traces, Dynamic Sigma"
	env = WindyGridworldEnv()
	num_episodes = 2000
	smoothing_window = 1
	stats_q_sigma_on_policy = q_sigma_on_policy_eligibility_traces_dynamic_sigma(env, num_episodes)
	rewards_stats_q_sigma_on_policy = pd.Series(stats_q_sigma_on_policy.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
	cum_rwd = rewards_stats_q_sigma_on_policy
	np.save('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Project_652/Code/Tabular/WindyGridWorld_Results/'  + 'Q_Sigma_On_Policy_Eligibility_Traces_Dynamic_Sigma' + '.npy', cum_rwd)
	# plotting.plot_episode_stats(stats_q_sigma_on_policy)
	env.close()



	print "Q(sigma) Off Policy Eligiblity Traces, Static Sigma"
	env = WindyGridworldEnv()
	num_episodes = 2000
	smoothing_window = 1
	stats_q_sigma_on_policy = q_sigma_off_policy_eligibility_traces_static_sigma(env, num_episodes)
	rewards_stats_q_sigma_on_policy = pd.Series(stats_q_sigma_on_policy.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
	cum_rwd = rewards_stats_q_sigma_on_policy
	np.save('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Project_652/Code/Tabular/WindyGridWorld_Results/'  + 'Q_Sigma_Off_Policy_Eligibility_Traces_Static_Sigma' + '.npy', cum_rwd)
	# plotting.plot_episode_stats(stats_q_sigma_on_policy)
	env.close()



	print "Q(sigma) Off Policy Eligiblity Traces, Dynamic Sigma"
	env = WindyGridworldEnv()
	num_episodes = 2000
	smoothing_window = 1
	stats_q_sigma_on_policy = q_sigma_off_policy_eligibility_traces_dynamic_sigma(env, num_episodes)
	rewards_stats_q_sigma_on_policy = pd.Series(stats_q_sigma_on_policy.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
	cum_rwd = rewards_stats_q_sigma_on_policy
	np.save('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Project_652/Code/Tabular/WindyGridWorld_Results/'  + 'Q_Sigma_Off_Policy_Eligibility_Traces_Dynamic_Sigma' + '.npy', cum_rwd)
	# plotting.plot_episode_stats(stats_q_sigma_on_policy)
	env.close()



if __name__ == '__main__':
	main()





