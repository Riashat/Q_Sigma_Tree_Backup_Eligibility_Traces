import copy
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd


eps = 1000
eps = range(eps)


off_policy_ac = np.load('Off_Policy_Actor_Critic_Rwd.npy')
static_sigma_off_ac = np.load('Unified_Static_Sigma_OffPolicy_AC_Rwd.npy')
decaying_sigma_off_ac = np.load('Unified_Decaying_Sigma_OffPolicy_AC_Rwd.npy')
count_expl_sigma_off_ac = np.load('Unified_Expl_Count_Sigma_OffPolicy_AC_Rwd.npy')
epsilon_expl_sigma_off_ac = np.load('Unified_Expl_Epsilon_Sigma_OffPolicy_AC_Rwd.npy')


on_policy_ac = np.load('On_Policy_Actor_Critic_Rwd.npy')
static_sigma_on_ac = np.load('Unified_Static_Sigma_OnPolicy_AC_Rwd.npy')
decaying_sigma_on_ac = np.load('Unified_Decay_Sigma_OnPolicy_AC_Rwd.npy')
count_expl_sigma_on_ac = np.load('Unified_Expl_Count_Sigma_OnPolicy_AC_Rwd.npy')
epsilon_expl_sigma_on_ac = np.load('Unified_Expl_Epsilon_Sigma_OnPolicy_AC_Rwd.npy')


def off_policy_comparison(stats1, stats2, stats3,  stats4, stats5, smoothing_window=1, noshow=False):

    fig = plt.figure(figsize=(30, 20))
    rewards_smoothed_1 = pd.Series(stats1).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_2 = pd.Series(stats2).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_3 = pd.Series(stats3).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_4 = pd.Series(stats4).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_5 = pd.Series(stats5).rolling(smoothing_window, min_periods=smoothing_window).mean()


    cum_rwd_1, = plt.plot(eps, rewards_smoothed_1, label="Off Policy Actor-Critic")    
    cum_rwd_2, = plt.plot(eps, rewards_smoothed_2, label="Static Sigma, Unified Actor-Critic")    
    cum_rwd_3, = plt.plot(eps, rewards_smoothed_3, label="Decaying Sigma, Unified Actor-Critic")    
    cum_rwd_4, = plt.plot(eps, rewards_smoothed_4, label="Count Exploration Sigma, Unified Actor-Critic")    
    cum_rwd_5, = plt.plot(eps, rewards_smoothed_5, label="Epsilon Exploration Sigma, Unified Actor-Critic")    


    plt.legend(handles=[cum_rwd_1, cum_rwd_2, cum_rwd_3,cum_rwd_4, cum_rwd_5])

    plt.xlabel("Epsiode")
    plt.ylabel("Epsiode Reward (Smoothed)")
    # plt.ylim((-200, 10))
    plt.title("Off-Policy Actor-Critic and Unified Actor-Critic - Windy Grid World Environment")  
    plt.show()

    return fig


def on_policy_comparison(stats1, stats2, stats3,  stats4, stats5, smoothing_window=1, noshow=False):

    fig = plt.figure(figsize=(30, 20))
    rewards_smoothed_1 = pd.Series(stats1).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_2 = pd.Series(stats2).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_3 = pd.Series(stats3).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_4 = pd.Series(stats4).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_5 = pd.Series(stats5).rolling(smoothing_window, min_periods=smoothing_window).mean()


    cum_rwd_1, = plt.plot(eps, rewards_smoothed_1, label="On Policy Actor-Critic")    
    cum_rwd_2, = plt.plot(eps, rewards_smoothed_2, label="Static Sigma, Unified Actor-Critic")    
    cum_rwd_3, = plt.plot(eps, rewards_smoothed_3, label="Decaying Sigma, Unified Actor-Critic")    
    cum_rwd_4, = plt.plot(eps, rewards_smoothed_4, label="Count Exploration Sigma, Unified Actor-Critic")    
    cum_rwd_5, = plt.plot(eps, rewards_smoothed_5, label="Epsilon Exploration Sigma, Unified Actor-Critic")    


    plt.legend(handles=[cum_rwd_1, cum_rwd_2, cum_rwd_3,cum_rwd_4, cum_rwd_5])

    plt.xlabel("Epsiode")
    plt.ylabel("Epsiode Reward (Smoothed)")
    # plt.ylim((-200, 10))
    plt.title("On-Policy Actor-Critic and Unified Actor-Critic - Windy Grid World Environment")  
    plt.show()

    return fig



eps = 300
eps = range(eps)


off_policy_ac = off_policy_ac[0:300]
static_sigma_off_ac = static_sigma_off_ac[0:300]
decaying_sigma_off_ac = decaying_sigma_off_ac[0:300]
count_expl_sigma_off_ac = count_expl_sigma_off_ac[0:300]
epsilon_expl_sigma_off_ac = epsilon_expl_sigma_off_ac[0:300]

on_policy_ac = on_policy_ac[0:300]
static_sigma_on_ac = static_sigma_on_ac[0:300]
decaying_sigma_on_ac = decaying_sigma_on_ac[0:300]
count_expl_sigma_on_ac = count_expl_sigma_on_ac[0:300]
epsilon_expl_sigma_on_ac = epsilon_expl_sigma_on_ac[0:300]





def main():

   off_policy_comparison(off_policy_ac, static_sigma_off_ac, decaying_sigma_off_ac, count_expl_sigma_off_ac, epsilon_expl_sigma_off_ac)
   #on_policy_comparison(on_policy_ac, static_sigma_on_ac, decaying_sigma_on_ac, count_expl_sigma_on_ac, epsilon_expl_sigma_on_ac)




if __name__ == '__main__':
	main()

