import matplotlib
import numpy as np
import pandas as pd
from collections import namedtuple
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])

def plot_cost_to_go_mountain_car(env, estimator, num_tiles=20):
    x = np.linspace(env.observation_space.low[0], env.observation_space.high[0], num=num_tiles)
    y = np.linspace(env.observation_space.low[1], env.observation_space.high[1], num=num_tiles)
    X, Y = np.meshgrid(x, y)
    Z = np.apply_along_axis(lambda _: -np.max(estimator.predict(_)), 2, np.dstack([X, Y]))

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                           cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Value')
    ax.set_title("Mountain \"Cost To Go\" Function")
    fig.colorbar(surf)
    plt.show()


def plot_value_function(V, title="Value Function"):
    """
    Plots the value function as a surface plot.
    """
    min_x = min(k[0] for k in V.keys())
    max_x = max(k[0] for k in V.keys())
    min_y = min(k[1] for k in V.keys())
    max_y = max(k[1] for k in V.keys())

    x_range = np.arange(min_x, max_x + 1)
    y_range = np.arange(min_y, max_y + 1)
    X, Y = np.meshgrid(x_range, y_range)

    # Find value for all (x, y) coordinates
    Z_noace = np.apply_along_axis(lambda _: V[(_[0], _[1], False)], 2, np.dstack([X, Y]))
    Z_ace = np.apply_along_axis(lambda _: V[(_[0], _[1], True)], 2, np.dstack([X, Y]))

    def plot_surface(X, Y, Z, title):
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                               cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
        ax.set_xlabel('Player Sum')
        ax.set_ylabel('Dealer Showing')
        ax.set_zlabel('Value')
        ax.set_title(title)
        ax.view_init(ax.elev, -120)
        fig.colorbar(surf)
        plt.show()

    plot_surface(X, Y, Z_noace, "{} (No Usable Ace)".format(title))
    plot_surface(X, Y, Z_ace, "{} (Usable Ace)".format(title))



def plot_episode_stats(stats, smoothing_window=200, noshow=False):
    # Plot the episode length over time
    fig1 = plt.figure(figsize=(10,5))
    plt.plot(stats.episode_lengths)
    plt.xlabel("Epsiode")
    plt.ylabel("Epsiode Length")
    plt.title("Episode Length over Time")
    if noshow:
        plt.close(fig1)
    else:
        plt.show(fig1)

    # Plot the episode reward over time
    fig2 = plt.figure(figsize=(10,5))
    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Epsiode")
    plt.ylabel("Epsiode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    if noshow:
        plt.close(fig2)
    else:
        plt.show(fig2)

    # Plot time steps and episode number
    fig3 = plt.figure(figsize=(10,5))
    plt.plot(np.cumsum(stats.episode_lengths), np.arange(len(stats.episode_lengths)))
    plt.xlabel("Time Steps")
    plt.ylabel("Episode")
    plt.title("Episode per time step")
    if noshow:
        plt.close(fig3)
    else:
        plt.show(fig3)

    return fig1, fig2, fig3


def plot_episode_multiple_stats(stats1, stats2, stats3, stats4, stats5,  smoothing_window=1, noshow=False):
    # Plot the episode length over time
    fig1 = plt.figure(figsize=(10,5))
    l1, = plt.plot(stats1.episode_lengths, label="SARSA")
    l2, = plt.plot(stats2.episode_lengths, label="Q Learning")
    l3, = plt.plot(stats3.episode_lengths, label="Expected SARSA")
    l4, = plt.plot(stats4.episode_lengths, label="Two Step Tree Backup")
    l5, = plt.plot(stats5.episode_lengths, label="Q(sigma)")

    plt.legend(handles=[l1, l2, l3, l4, l5])
    plt.xlabel("Epsiode")
    plt.ylabel("Epsiode Length")
    plt.title("Comparing TD Algorithms - Episode Length over Time")
    if noshow:
        plt.close(fig1)
    else:
        plt.show(fig1)

    # Plot the episode reward over time
    fig2 = plt.figure(figsize=(10,5))
    rewards_smoothed1 = pd.Series(stats1.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed2 = pd.Series(stats2.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed3 = pd.Series(stats3.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed4 = pd.Series(stats4.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed5 = pd.Series(stats5.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()


    cum_rwd_1, = plt.plot(rewards_smoothed1, label="SARSA")
    cum_rwd_2, = plt.plot(rewards_smoothed2, label="Q Learning")
    cum_rwd_3, = plt.plot(rewards_smoothed3, label="Expected SARSA")
    cum_rwd_4, = plt.plot(rewards_smoothed4, label="2-Step Tree Backup")
    cum_rwd_5, = plt.plot(rewards_smoothed5, label="Q(sigma)")

    plt.legend(handles=[cum_rwd_1, cum_rwd_2, cum_rwd_3, cum_rwd_4, cum_rwd_5])
    plt.xlabel("Epsiode")
    plt.ylabel("Epsiode Reward (Smoothed)")
    plt.title('Comparing TD Algorithms - Q(sigma) and Tree Backup')
    if noshow:
        plt.close(fig2)
    else:
        plt.show(fig2)


    # Plot time steps and episode number
    fig3 = plt.figure(figsize=(10,5))
    t1, = plt.plot(np.cumsum(stats1.episode_lengths), np.arange(len(stats1.episode_lengths)), label="SARSA")
    t2, = plt.plot(np.cumsum(stats2.episode_lengths), np.arange(len(stats2.episode_lengths)), label="Q Learning")
    t3, = plt.plot(np.cumsum(stats3.episode_lengths), np.arange(len(stats3.episode_lengths)), label="Expected SARSA")
    t4, = plt.plot(np.cumsum(stats4.episode_lengths), np.arange(len(stats4.episode_lengths)), label="2 Step Tree Backup")
    t5, = plt.plot(np.cumsum(stats5.episode_lengths), np.arange(len(stats5.episode_lengths)), label="Q(sigma)")

    plt.legend(handles=[t1, t2, t3, t4, t5])
    plt.xlabel("Time Steps")
    plt.ylabel("Episode")
    plt.title("Episode per time step")
    if noshow:
        plt.close(fig3)
    else:
        plt.show(fig3)

    return fig1, fig2, fig3



