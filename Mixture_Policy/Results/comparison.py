import copy
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd


eps = 500
eps = range(eps)


mix_count = np.load('mixture_actor_critic_countexpl_rwd.npy')
mix_eps = np.load('mixture_actor_critic_epsilonexpl_rwd.npy')

naive_on = np.load('On_Policy_Actor_Critic_Rwd.npy')
naive_off = np.load('Off_Policy_Actor_Critic_Rwd.npy')


off_count = np.load('Unified_Expl_Count_Sigma_OffPolicy_AC_Rwd.npy')
off_eps = np.load('Unified_Expl_Epsilon_Sigma_OffPolicy_AC_Rwd.npy')

on_count = np.load('Unified_Expl_Count_Sigma_OnPolicy_AC_Rwd.npy')
on_eps = np.load('Unified_Expl_Epsilon_Sigma_OnPolicy_AC_Rwd.npy')






def comparison(stats1, stats2, stats3,  stats4, stats5, stats6, smoothing_window=50, noshow=False):

    fig = plt.figure(figsize=(30, 20))
    rewards_smoothed_1 = pd.Series(stats1).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_2 = pd.Series(stats2).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_3 = pd.Series(stats3).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_4 = pd.Series(stats4).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_5 = pd.Series(stats5).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_6 = pd.Series(stats6).rolling(smoothing_window, min_periods=smoothing_window).mean()

    cum_rwd_1, = plt.plot(eps, rewards_smoothed_1, label="Unifying On-Off Policy Count Expl Sigma")    
    cum_rwd_2, = plt.plot(eps, rewards_smoothed_2, label="Unifying On-Off Policy Epsilon Expl Sigma")    
    cum_rwd_3, = plt.plot(eps, rewards_smoothed_3, label="Off Policy Count Expl Sigma")    
    cum_rwd_4, = plt.plot(eps, rewards_smoothed_4, label="Off Policy Epsilon Expl Sigma")    
    cum_rwd_5, = plt.plot(eps, rewards_smoothed_5, label="On Policy Count Expl Sigma")    
    cum_rwd_6, = plt.plot(eps, rewards_smoothed_6, label="On Policy Epsilon Expl Sigma")    


    plt.legend(handles=[cum_rwd_1, cum_rwd_2, cum_rwd_3,cum_rwd_4, cum_rwd_5, cum_rwd_6])

    plt.xlabel("Epsiode")
    plt.ylabel("Epsiode Reward (Smoothed)")
    # plt.ylim((-200, 10))
    plt.title("Unifying On Policy and Off Policy - Actor Critic Methods")  
    plt.show()

    return fig




def comparison_mixture_naive(stats1, stats2, stats3,  stats4, smoothing_window=10, noshow=False):

    fig = plt.figure(figsize=(30, 20))
    rewards_smoothed_1 = pd.Series(stats1).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_2 = pd.Series(stats2).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_3 = pd.Series(stats3).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_4 = pd.Series(stats4).rolling(smoothing_window, min_periods=smoothing_window).mean()


    cum_rwd_1, = plt.plot(eps, rewards_smoothed_1, label="Unifying On-Off Policy Count Expl Sigma")    
    cum_rwd_2, = plt.plot(eps, rewards_smoothed_2, label="Unifying On-Off Policy Epsilon Expl Sigma")    
    cum_rwd_3, = plt.plot(eps, rewards_smoothed_3, label="On-Policy Actor-Critic")    
    cum_rwd_4, = plt.plot(eps, rewards_smoothed_4, label="Off-Policy Actor-Critic")    



    plt.legend(handles=[cum_rwd_1, cum_rwd_2, cum_rwd_3,cum_rwd_4])

    plt.xlabel("Epsiode")
    plt.ylabel("Epsiode Reward (Smoothed)")
    # plt.ylim((-200, 10))
    plt.title("Comparing Actor-Critic with Unified On-Policy and Off-Policy Actor-Critic")  
    plt.show()

    return fig




eps = 500
eps = range(eps)




mix_count = np.load('mixture_actor_critic_countexpl_rwd.npy')
mix_eps = np.load('mixture_actor_critic_epsilonexpl_rwd.npy')

naive_on = np.load('On_Policy_Actor_Critic_Rwd.npy')
naive_off = np.load('Off_Policy_Actor_Critic_Rwd.npy')

naive_on = naive_on[0:500]
naive_off = naive_off[0:500]






def main():


   # comparison(mix_count, mix_eps, off_count, off_eps, on_count, on_eps)
   comparison_mixture_naive(mix_count, mix_eps, naive_off, naive_on)



if __name__ == '__main__':
	main()


