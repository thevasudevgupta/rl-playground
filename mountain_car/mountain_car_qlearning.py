#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 02:17:13 2020

@author: vasudevgupta
"""

import gym
import time
import numpy as np
env= gym.make('MountainCar-v0')

print(env.observation_space)
print(env.action_space)

## descretization of states
def descretize_state(state):
    new_state= (state- env.observation_space.low)*np.array([10, 100])
    new_state= np.round(new_state, 0).astype(int)
    return new_state

def Q_model(env, episodes= 5000, learning_rate= 0.2, discount= 0.9, ep= 0.8, min_ep= 0):
    
    ## we have 2 states; both are continuous; either descretize them or use Deep Q networks
    num_actions= env.action_space.n ## actions can be 0,1,2
    num_states= (env.observation_space.high- env.observation_space.low)*np.array([10, 100])
    num_states= np.round(num_states).astype(int) + 1
    Q_table= np.random.rand(num_states[0], num_states[1], num_actions)
    redn= (ep-min_ep)/episodes
    treward_episode= []
    
    start = time.time()
    for i in range(episodes):
        init_state= env.reset()
        init_state= descretize_state(init_state)
        done= False
        total_reward= 0
        while not done:
            if i >= (episodes - 20):
                env.render()
                
            if np.random.random() < 1- ep:
                action= np.argmax(Q_table[init_state[0], init_state[1], :])
            else: ## lets do some exploration using greedy epsilon strategy
                action= np.random.randint(0, num_actions)
                
            final_state, reward, done, _ = env.step(action)
            new_state= descretize_state(final_state)
            
            if done and final_state[0]>= 0.5:
                # episode is over
                Q_table[init_state[0], init_state[1], action]= reward
            else: ## Q value will be updated as per bellsman eqn
                delta = learning_rate*(reward + 
                                 discount*np.max(Q_table[new_state[0], 
                                                   new_state[1]]) - 
                                 Q_table[init_state[0], init_state[1],action])
                Q_table[init_state[0], init_state[1],action] += delta
                
            total_reward+= reward
            init_state= new_state
            
        if ep > min_ep:
            ep -= redn
        
        treward_episode.append(total_reward)
    print(time.time()- start)
    return treward_episode
    
## lets run this model
rewards= Q_model(env, episodes= 5000)
env.close()
     