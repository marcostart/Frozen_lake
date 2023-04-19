#!/usr/bin/env python3

import gymnasium as gym
import numpy as np
import os
import pickle


env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=True, render_mode="rgb_array")

def init_q(s, a): 
            """
            s: number of states
            a: number of actions
            """
            return np.zeros((s, a))

      # epsilon-greedy exploration strategy
def epsilon_greedy(n_actions, Q, epsilon, s):
    """
    Q: Q Table
    epsilon: exploration parameter
    s: state
    """
    # selects a random action with probability epsilon
    if np.random.random() <= epsilon:
            return np.random.randint(n_actions)
    else:
            return np.argmax(Q[s, :])

# SARSA Process
def train_sarsa(env, alpha, gamma, start_epsilon, min_epsilon, decay_rate, n_episodes, n_states, n_actions): 
    """
    alpha: learning rate
    gamma: exploration parameter
    n_episodes: number of episodes
    """
    # initialize Q table
    Q = init_q(n_states, n_actions)
    # to record reward for each episode
    reward_array = np.zeros(n_episodes)
    for i in range(n_episodes):
            # initial state
            print("episode :", i+1)
            env.reset()
            s= env.s
            epsilon = max(min_epsilon, (start_epsilon - min_epsilon)*np.exp(-decay_rate*i))
            print("epsilon : ", epsilon)
            # initial action
            a = epsilon_greedy(n_actions, Q, epsilon, s)
            done = False
            while not done:
                s_, reward, done, truncated, info  = env.step(a)
                
                a_ = epsilon_greedy(n_actions, Q, epsilon, s_)
                # update Q table
                Q[s, a] = (1 - alpha) * Q[s, a] + alpha * (reward + gamma * Q[s_, a_])
                # update processing bar
                if done:
                        reward_array[i] = reward
                        break
                s, a = s_, a_
    env.close()
    # show Q table
    print('Trained Q Table:')
    print(Q)
    # show average reward
    avg_reward = round(np.mean(reward_array), 4)
    print('Training Averaged reward per episode {}'.format(avg_reward))
    return Q



observation, info = env.reset(seed=42)
state_obs_space = env.observation_space # Returns sate(observation) space of the environment.
action_space = env.action_space # Returns action space of the environment.
print("State(Observation) space:", state_obs_space)
print("Action space:", action_space)

n_actions = env.action_space.n
n_states = env.observation_space.n

if not os.path.exists('./save_frozen_lake_slip.pkl') :
    # SARSA parameters
    alpha = 0.1   # learning rate
    gamma = 0.95  # discount factor

    # Training parameters
    n_episodes = 1000000  # number of episodes to use for training
    n_max_steps = 100   # maximum number of steps per episode

    # Exploration / Exploitation parameters
    start_epsilon = 1.0  # start training by selecting purely random actions
    min_epsilon = 0.0   # the lowest epsilon allowed to decay to
    decay_rate = 0.00001   # epsilon will gradually decay so we do less exploring and more exploiting as Q-function improves

    Q = train_sarsa(env, alpha, gamma, start_epsilon, min_epsilon, decay_rate, n_episodes, n_states, n_actions)
    fd = open('save_frozen_lake_slip.pkl', 'wb')
    pickle.dump(Q, fd, -1)


env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=True, render_mode="human")

env.reset()
saved_fd = open('save_frozen_lake_slip.pkl', 'rb')
Q = pickle.load(saved_fd)
state = env.s
done = False
while not done:
    action = np.argmax(Q[state, :])
    state, reward, done,truncated, info = env.step(action)
    env.render()
    if (truncated):
        done = True
