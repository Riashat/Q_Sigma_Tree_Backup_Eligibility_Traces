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


# from collections import defaultdict
# from lib.envs.cliff_walking import CliffWalkingEnv
# from lib import plotting
# env = CliffWalkingEnv()



# from collections import defaultdict
# from lib.envs.gridworld import GridworldEnv
# from lib import plotting
# env = GridworldEnv()


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



def sarsa_2_step_TD(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):

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
			next_state, reward, _, _ = env.step(action)
			next_action_probs = policy(next_state)
			next_action = np.random.choice(np.arange(len(action_probs)), p = next_action_probs)


			next_2_state, reward_2_step, done, _ = env.step(next_action)
			next_2_action_probs = policy(next_2_state)
			next_2_action = np.random.choice(np.arange(len(next_2_action_probs)), p = next_2_action_probs)


			# stats.episode_rewards[i_episode] += reward
			# stats.episode_lengths[i_episode] = t

			stats.episode_rewards[i_episode] += reward_2_step
			stats.episode_lengths[i_episode] = t


			# TD Update Equations
			#TD Target - One step ahead
			td_target = reward + discount_factor * reward_2_step + discount_factor*discount_factor * Q[next_2_state][next_2_action]
			
			# TD Error
			td_delta = td_target - Q[state][action]

			Q[state][action] += alpha * td_delta

			
			if done:
				break
			action = next_action
			state = next_state

	return Q, stats


def q_learning_2_step_TD(env, num_episodes, discount_factor=1.0, alpha = 0.5, epsilon = 0.1):

	Q = defaultdict(lambda : np.zeros(env.action_space.n))

	stats = plotting.EpisodeStats(episode_lengths=np.zeros(num_episodes),episode_rewards=np.zeros(num_episodes))  

	#policy that the agent is following
	policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)


	for i_episode in range(num_episodes):

		state = env.reset()

		for t in itertools.count():

			action_probs = policy(state)
			action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
			next_state, reward, done, _, = env.step(action)


			next_2_state, reward_2_step, done, _ = env.step(action)
			next_2_action_probs = policy(next_2_state)
			next_2_action = np.random.choice(np.arange(len(next_2_action_probs)), p = next_2_action_probs)


			# stats.episode_rewards[i_episode] += reward
			# stats.episode_lengths[i_episode] = t

			stats.episode_rewards[i_episode] += reward_2_step
			stats.episode_lengths[i_episode] = t

			# TD Update Equations:


			best_next_action = np.argmax(Q[next_2_state])

			td_target = reward +  discount_factor * reward_2_step +   discount_factor * discount_factor * Q[next_2_state][best_next_action]

			td_delta = td_target - Q[state][action]

			#update Q function based on the TD error
			Q[state][action] += alpha * td_delta

			if done:
				break

			state = next_state


	return Q, stats


def q_learning_4_step_TD(env, num_episodes, discount_factor=1.0, alpha = 0.5, epsilon = 0.1):



	Q = defaultdict(lambda : np.zeros(env.action_space.n))

	stats = plotting.EpisodeStats(episode_lengths=np.zeros(num_episodes),episode_rewards=np.zeros(num_episodes))  

	#policy that the agent is following
	policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)


	for i_episode in range(num_episodes):

		state = env.reset()

		for t in itertools.count():

			action_probs = policy(state)
			action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
			next_state, reward, done, _, = env.step(action)


			next_2_state, reward_2_step, done, _ = env.step(action)
			next_2_action_probs = policy(next_2_state)
			next_2_action = np.random.choice(np.arange(len(next_2_action_probs)), p = next_2_action_probs)
			# stats.episode_rewards[i_episode] += reward_2_step
			# stats.episode_lengths[i_episode] = t

			next_3_state, reward_3_step, done, _ = env.step(next_2_action)
			next_3_action_probs = policy(next_3_state)
			next_3_action = np.random.choice(np.arange(len(next_3_action_probs)), p = next_3_action_probs)
			# stats.episode_rewards[i_episode] += reward_3_step
			# stats.episode_lengths[i_episode] = t

			next_4_state, reward_4_step, done, _ = env.step(next_3_action)
			next_4_action_probs = policy(next_4_state)
			next_4_action = np.random.choice(np.arange(len(next_4_action_probs)), p = next_4_action_probs)
			stats.episode_rewards[i_episode] += reward_4_step
			stats.episode_lengths[i_episode] = t



			# TD Update Equations:
			best_next_action = np.argmax(Q[next_4_state])

			td_target = reward +  discount_factor * reward_2_step +  discount_factor * discount_factor * reward_3_step + discount_factor * discount_factor * discount_factor *reward_4_step +  discount_factor * discount_factor * discount_factor * discount_factor * Q[next_4_state][best_next_action]

			td_delta = td_target - Q[state][action]

			#update Q function based on the TD error
			Q[state][action] += alpha * td_delta

			if done:
				break

			state = next_state


	return Q, stats


"""
Two Step Tree Backup
"""

def two_step_tree_backup(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):


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
            next_state, reward, _ , _ = env.step(action)
            
            #reward by taking action under the policy pi
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            next_action_probs = policy(next_state)
            next_action = np.random.choice(np.arange(len(next_action_probs)), p =next_action_probs )

            #V = sum_a pi(a, s_{t+1})Q(s_{t+1}, a)
            V = np.sum(next_action_probs * Q[next_state])


            next_next_state, next_reward, done, _ = env.step(next_action)
    
            next_next_action_probs = policy(next_next_state)
            next_next_action = np.random.choice(np.arange(len(next_next_action_probs)), p = next_next_action_probs)

            next_V = np.sum(next_next_action_probs * Q[next_next_state])            


            # print "Next Action:", next_action
            # print "Next Action probs :", next_action_probs

            #Main Update Equations for Two Step Tree Backup
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

    return Q, stats


"""
Three Step Tree Backup
"""


def three_step_tree_backup(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):

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
            next_state, reward, _ , _ = env.step(action)

            next_action_probs = policy(next_state)
            next_action = np.random.choice(np.arange(len(next_action_probs)), p =next_action_probs )
            next_next_state, next_reward, _, _ = env.step(next_action)
    

            next_next_action_probs = policy(next_next_state)
            next_next_action = np.random.choice(np.arange(len(next_next_action_probs)), p = next_next_action_probs)
            next_next_next_state, next_next_reward, done, _ = env.step(next_next_action)
 
            next_next_next_action_probs  = policy(next_next_next_state)
            next_next_next_action = np.random.choice(np.arange(len(next_next_next_action_probs)), p = next_next_next_action_probs)

 

            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            #updates for the Three Step Tree Backup

            #V = sum_a pi(a, s_{t+1})Q(s_{t+1}, a)
            V = np.sum(next_action_probs * Q[next_state])

            One_Step = reward + discount_factor * V

            next_V = np.sum(next_next_action_probs * Q[next_next_state])            
            Delta_1 = next_reward + discount_factor * next_V - Q[next_state][next_action]
            next_action_selection_probability = np.max(next_action_probs)            

            Two_Step = discount_factor * next_action_selection_probability * Delta_1

            next_next_V = np.sum(next_next_next_action_probs * Q[next_next_next_state])
            Delta_2 = next_next_reward + discount_factor * next_next_V - Q[next_next_state][next_next_action]
            next_next_action_selection_probability = np.max(next_next_action)

            Three_Step = discount_factor * next_action_selection_probability * discount_factor * next_next_action_selection_probability * Delta_2


            td_target = One_Step + Two_Step + Three_Step 

            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta


            if done:
                break

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



def plot_episode_stats(stats1, stats2, stats3, stats4, stats5, stats6, stats7, smoothing_window=200, noshow=False):

	#higher the smoothing window, the better the differences can be seen

    # Plot the episode reward over time
    fig = plt.figure(figsize=(20, 10))
    rewards_smoothed_1 = pd.Series(stats1.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_2 = pd.Series(stats2.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_3 = pd.Series(stats3.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_4 = pd.Series(stats4.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_5 = pd.Series(stats5.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_6 = pd.Series(stats6.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_7 = pd.Series(stats7.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()


    cum_rwd_1, = plt.plot(rewards_smoothed_1, label="2-Step SARSA")
    cum_rwd_2, = plt.plot(rewards_smoothed_2, label="2-Step Q Learning")
    cum_rwd_3, = plt.plot(rewards_smoothed_3, label="4-Step Q Learning")
    cum_rwd_4, = plt.plot(rewards_smoothed_4, label="2-Step Tree Backup")
    cum_rwd_5, = plt.plot(rewards_smoothed_5, label="3-Step Tree Backup")
    cum_rwd_6, = plt.plot(rewards_smoothed_6, label="Q Learning")
    cum_rwd_7, = plt.plot(rewards_smoothed_7, label="SARSA")


    plt.legend(handles=[cum_rwd_1, cum_rwd_2, cum_rwd_3, cum_rwd_4, cum_rwd_5, cum_rwd_6, cum_rwd_7])
    plt.xlabel("Epsiode")
    plt.ylabel("Epsiode Reward (Smoothed)")
    plt.title("Comparing Multi-Step TD Learning Algorithms on Grid World Environment")
    plt.show()


    return fig



def main():

	Number_Episodes = 1500
	print "SARSA 2 Step"
	SARSA_2_Step, stats_sarsa_2_step = sarsa_2_step_TD(env, Number_Episodes)
	
	print "Q-Learning 2 Step"
	Q_Learning_2_Step, stats_q_learning_2_step= q_learning_2_step_TD(env, Number_Episodes)
	
	print "Q Learning 4 Step"
	Q_Learning_4_Step, stats_q_learning_4_step = q_learning_4_step_TD(env, Number_Episodes)

	print "Tree Backup 2 Step"
	TB_2_Step, stats_TB_2_step = two_step_tree_backup(env, Number_Episodes)

	print "Tree Backup 3 Step"
	TB_3_Step, stats_TB_3_step = three_step_tree_backup(env, Number_Episodes)

	print "Q Learning"
	Q_learn, stats_q_learning = q_learning(env, Number_Episodes)

	print "SARSA"
	SARSA_1, stats_sarsa = sarsa(env, Number_Episodes)



	plot_episode_stats(stats_sarsa_2_step, stats_q_learning_2_step, stats_q_learning_4_step, stats_TB_2_step, stats_TB_3_step, stats_q_learning, stats_sarsa)




if __name__ == '__main__':
	main()
