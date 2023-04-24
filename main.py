#!/usr/bin/env python

# importer les bibliothèques
import gymnasium as gym
import numpy as np
import pickle as pkl
import random
import os

# Créer l'environnement 
env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=False, render_mode="rgb_array").env

if not os.path.exists('./frozen_lake_1.pkl') :
    # Définition des paramètres
    alpha = 0.2
    gamma = 0.8
    epsilon = 0.9
    max_episodes = 800000


    # Initialiser la Q-Matrice
    Q = np.zeros((env.observation_space.n, env.action_space.n))

    # Début de l'apprentissage avec SARSA
    for episode in range(max_episodes):
        print('Episode :', episode)
        env.reset()
        done = False
        state1 = env.s
        if random.uniform(0,1) < epsilon:
            action1 = env.action_space.sample()
        else:
            action1 = np.argmax(Q[state1])

        while not done:
            # Prochain State (Etat)
            state2, reward, is_terminated, is_truncated, info = env.step(action1)
        
            # Choisir la prochaine action
            if random.uniform(0,1) < epsilon:
                action2 = env.action_space.sample()
            else:
                action2 = np.argmax(Q[state2])
        
            # Actualiser la Q-Matrice
            Q[state1, action1] = (1 - alpha) * Q[state1, action1] + alpha * (reward + gamma * Q[state2, action2])
        
            # Actualiser les paramètres 
            state1 = state2
            action1 = action2
        
            # Fin de l'épisode
            if (is_terminated or is_truncated):
                done = True
                break
    fd = open('frozen_lake_1.pkl', 'wb')
    pkl.dump(Q, fd, -1)



env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=False, render_mode="human").env
env.reset()

saved_fd = open('frozen_lake_1.pkl', 'rb')
Q = pkl.load(saved_fd)
is_terminated = False
is_truncated = False
state = env.s

while 1 :
    if (not is_terminated) and (not is_truncated) :
        action = np.argmax(Q[state])
        next_state, reward, is_terminated, is_truncated, info = env.step(action)
        state = next_state
    else :
        break
        