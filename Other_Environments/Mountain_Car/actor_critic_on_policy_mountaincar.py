import sys
sys.path.insert(0, "/Users/Riashat/Documents/PhD_Research/Tree_Backup_Q_Sigma_Function_Approximation/Linear_Approximator/")
import gym
import itertools
import matplotlib
import numpy as np
import sys
import tensorflow as tf
import collections

import sklearn.pipeline
import sklearn.preprocessing

from collections import defaultdict
from lib.envs.cliff_walking import CliffWalkingEnv
from lib.envs.windy_gridworld import WindyGridworldEnv
from lib import plotting

from sklearn.kernel_approximation import RBFSampler

from lib.envs.cliff_walking import CliffWalkingEnv
from lib import plotting
matplotlib.style.use('ggplot')

env = gym.envs.make("MountainCar-v0")
env.observation_space.sample()


# Feature Preprocessing: Normalize to zero mean and unit variance
# We use a few samples from the observation space to do this
observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(observation_examples)

# Used to converte a state to a featurizes represenation.
# We use RBF kernels with different variances to cover different parts of the space
featurizer = sklearn.pipeline.FeatureUnion([
        ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
        ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
        ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
        ("rbf4", RBFSampler(gamma=0.5, n_components=100))
        ])
featurizer.fit(scaler.transform(observation_examples))


def featurize_state(state):
    """
    Returns the featurized representation for a state.
    """
    # scaled = scaler.transform([state])
    scaled = scaler.transform(state)
    featurized = featurizer.transform(scaled)
    return featurized[0]





class PolicyEstimator():
    """
    Policy Function approximator. 
    """
    
    def __init__(self, learning_rate=0.01, scope="policy_estimator"):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.int32, [], "state")
            self.action = tf.placeholder(dtype=tf.int32, name="action")
            self.target = tf.placeholder(dtype=tf.float32, name="target")

            # This is just table lookup estimator
            # state_one_hot = featurize_state(self.state)
            # state_one_hot = tf.one_hot(self.state, int(env.observation_space))
            # self.output_layer = tf.contrib.layers.fully_connected(
            #     inputs=tf.expand_dims(state_one_hot, 0),
            #     num_outputs=env.action_space.n,
            #     activation_fn=None,
            #     weights_initializer=tf.zeros_initializer())


            # This is just linear classifier
            # self.state = featurize_state()
            self.output_layer = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(self.state, 0),
                num_outputs=env.action_space.n,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer())

            self.action_probs = tf.squeeze(tf.nn.softmax(self.output_layer))
            self.picked_action_prob = tf.gather(self.action_probs, self.action)

            # Loss and train op
            self.loss = -tf.log(self.picked_action_prob) * self.target

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())
    
    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        state = featurize_state(state)
        return sess.run(self.action_probs, { self.state: state })

    def update(self, state, target, action, sess=None):
        sess = sess or tf.get_default_session()
        state = featurize_state(state)
        feed_dict = { self.state: state, self.target: target, self.action: action  }
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss



class ValueEstimator():
    """
    Value Function approximator. 
    """
    
    def __init__(self, learning_rate=0.1, scope="value_estimator"):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.int32, [], "state")
            self.target = tf.placeholder(dtype=tf.float32, name="target")

            # This is just table lookup estimator
            state_one_hot = tf.one_hot(self.state, int(env.observation_space))
            self.output_layer = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(state_one_hot, 0),
                num_outputs=env.action_space.n,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer())

            self.value_estimate = tf.squeeze(self.output_layer)
            self.loss = tf.squared_difference(self.value_estimate, self.target)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())        
    
    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.value_estimate, { self.state: state })

    def update(self, state, target, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = { self.state: state, self.target: target }
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss



def actor_critic(env, estimator_policy, estimator_value, num_episodes, discount_factor=1.0):
    """
    Actor Critic Algorithm. Optimizes the policy 
    function approximator using policy gradient.
    
    Args:
        env: OpenAI environment.
        estimator_policy: Policy Function to be optimized 
        estimator_value: Value function approximator, used as a baseline
        num_episodes: Number of episodes to run for
        discount_factor: Time-discount factor
    
    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))    
  

    cumulative_errors = np.zeros(shape=(num_episodes, 1))

    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
    
    for i_episode in range(num_episodes):
        # Reset the environment and pick the fisrst action

        print "Number of Episodes", i_episode

        state = env.reset()
        
        episode = []
        
        # One step in the environment
        for t in itertools.count():
            
            # Take a step
            print X
            action_probs = estimator_policy.predict(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(action)
            
            # Keep track of the transition
            episode.append(Transition(
              state=state, action=action, reward=reward, next_state=next_state, done=done))
            
            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t
         
            next_action_probs = estimator_policy.predict(next_state)
            next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs)

            # Calculate TD Target
            value = estimator_value.predict(next_state)
            value_next = value[next_action]

            td_target = reward + discount_factor * value_next
            td_error = td_target - estimator_value.predict(state)
            rms_error = np.sqrt(np.sum((td_error)**2))
            cumulative_errors[i_episode, :] += rms_error


            # Update the value estimator and policy estimator
            estimator_value.update(state, td_target)
            
            estimator_policy.update(state, td_error, action)
            

            if done:
                break
                
            state = next_state
    
    return stats, cumulative_errors





def take_average_results(experiment,num_experiments,num_episodes,env,policy_estimator, value_estimator):

    reward_mat=np.zeros([num_episodes,num_experiments])
    error_mat=np.zeros([num_episodes,num_experiments])

    for i in range(num_experiments):

        stats,cum_error = experiment(env, value_estimator,  policy_estimator, num_episodes)
        reward_mat[:,i] = stats.episode_rewards
        error_mat[:,i] = cum_error.T
        average_reward = np.mean(reward_mat,axis=1)
        average_error = np.mean(error_mat,axis=1)

        np.save('/Users/Riashat/Documents/PhD_Research/Tree_Backup_Q_Sigma_Function_Approximation/Actor_Critic_Q_sigma/Results/'  + 'On_Policy_Actor_Critic_Rwd' + '.npy', average_reward)
        np.save('/Users/Riashat/Documents/PhD_Research/Tree_Backup_Q_Sigma_Function_Approximation/Actor_Critic_Q_sigma/Results/'  + 'On_Policy_Actor_Critic_Err' + '.npy', average_error)
        
    return average_reward,average_error



def main():
    tf.reset_default_graph()

    global_step = tf.Variable(0, name="global_step", trainable=False)
    policy_estimator = PolicyEstimator()
    value_estimator = ValueEstimator()

    num_episodes=1000
    num_experiments = 20

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        avg_cum_reward, avg_cum_error = take_average_results(actor_critic, num_experiments, num_episodes, env, value_estimator, policy_estimator)
    
    env.close()


if __name__ == '__main__':
    main()

