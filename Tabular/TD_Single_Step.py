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


# from lib.envs.cliff_walking import CliffWalkingEnv
# from lib import plotting
# env = CliffWalkingEnv()

from lib.envs.gridworld import GridworldEnv
from lib import plotting
env = GridworldEnv()

# from lib.envs.windy_gridworld import WindyGridworldEnv
# from lib import plotting
# env = WindyGridworldEnv()

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


	return Q, stats



def q_learning(env, num_episodes, discount_factor=1.0, alpha = 0.5, epsilon = 0.1):

	#Off Policy TD - Find Optimal Greedy policy while following epsilon-greedy policy

	Q = defaultdict(lambda : np.zeros(env.action_space.n))

	stats = plotting.EpisodeStats(episode_lengths=np.zeros(num_episodes),episode_rewards=np.zeros(num_episodes))  

	#policy that the agent is following
	policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)


	for i_episode in range(num_episodes):

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


	return Q, stats




def double_q_learning(env, num_episodes, discount_factor=1.0, alpha = 0.5, epsilon = 0.1):

	#Off Policy TD - Find Optimal Greedy policy while following epsilon-greedy policy

	Q_A = defaultdict(lambda : np.zeros(env.action_space.n))

	Q_B = defaultdict(lambda : np.zeros(env.action_space.n))

	Total_Q = defaultdict(lambda : np.zeros(env.action_space.n))

	stats = plotting.EpisodeStats(episode_lengths=np.zeros(num_episodes),episode_rewards=np.zeros(num_episodes))  

	#choose a based on Q_A + Q_B
	policy = make_epsilon_greedy_policy(Total_Q, epsilon, env.action_space.n)


	for i_episode in range(num_episodes):

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


	return Total_Q, stats



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

	return Q, stats



"""
Q(sigma) algorithm
"""

def q_sigma_on_policy(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.9):

    #Expected SARSA : same algorithm steps as Q-Learning, 
    # only difference : instead of maximum over next state and action pairs
    # use the expected value
    Q = defaultdict(lambda : np.zeros(env.action_space.n))
    stats = plotting.EpisodeStats(episode_lengths=np.zeros(num_episodes),episode_rewards=np.zeros(num_episodes))  

    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    for i_episode in range(num_episodes):

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



            if done:
                break
            action = next_action
            state = next_state

    return Q, stats





def plot_episode_stats(stats1, stats2, stats3, stats4, stats5, smoothing_window=200, noshow=False):

	#higher the smoothing window, the better the differences can be seen

    # Plot the episode reward over time
    fig = plt.figure(figsize=(20, 10))
    rewards_smoothed_1 = pd.Series(stats1.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_2 = pd.Series(stats2.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_3 = pd.Series(stats3.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_4 = pd.Series(stats4.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_5 = pd.Series(stats5.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()



    cum_rwd_1, = plt.plot(rewards_smoothed_1, label="SARSA")
    cum_rwd_2, = plt.plot(rewards_smoothed_2, label="Q Learning")
    cum_rwd_3, = plt.plot(rewards_smoothed_3, label="Double Q Learning")
    cum_rwd_4, = plt.plot(rewards_smoothed_4, label="Expected SARSA")
    cum_rwd_5, = plt.plot(rewards_smoothed_5, label="Q(sigma)")


    plt.legend(handles=[cum_rwd_1, cum_rwd_2, cum_rwd_3, cum_rwd_4, cum_rwd_5])
    plt.xlabel("Epsiode")
    plt.ylabel("Epsiode Reward (Smoothed)")
    plt.title("Comparison of TD Learning Algorithms on Cliff Walking Environment")
    plt.show()


    return fig



def main():

	Number_Episodes = 1500

	print "SARSA"
	Sarsa, stats_sarsa= sarsa(env, Number_Episodes)
	rewards_stats_sarsa = pd.Series(stats_sarsa.episode_rewards).rolling(100, min_periods=100).mean()


	print "Q-Learning"
	Q_learning, stats_q_learning= q_learning(env, Number_Episodes)
	
	print "Double Q Learning"
	Double_Q_Learning, stats_double_q_learning = double_q_learning(env, Number_Episodes)

	print "Expected SARSA"
	Expected_SARSA, stats_expected_sarsa = one_step_tree_backup(env, Number_Episodes)

	print "Q-Sigma"
	Q_Sigma, stats_q_sigma = q_sigma_on_policy(env, Number_Episodes, discount_factor=1.0, alpha = 0.5, epsilon = 0.9)



	plot_episode_stats(stats_sarsa, stats_q_learning, stats_double_q_learning, stats_expected_sarsa, stats_q_sigma)




if __name__ == '__main__':
	main()






