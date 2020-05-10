#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 02:43:18 2020

@author: vasudevgupta
"""

import gym
import tensorflow as tf
from collections import deque
import numpy as np
import random
import time
import matplotlib.pyplot as plt

env= gym.make('MsPacman-v0')

class Agent(object):
    def __init__(self):
        self.state_size= (124, 124)
        self.gamma= 0.9
        self.replay_memory= deque(maxlen= 10000)
        self.min_ep= 0.1
        self.num_actions= env.action_space.n
        self.qmodel1= self.conv_qnetwork()
        self.qmodel1.compile(optimizer= tf.keras.optimizers.Adam(LEARNING_RATE), loss= 'mse')
        self.target_model1= self.conv_qnetwork()
         
    def conv_qnetwork(self):
        model= tf.keras.Sequential([
             tf.keras.layers.Conv2D(filters= 8, kernel_size= (3,3), strides= (2,2), padding= "same", input_shape= (124, 124, 3)),
             tf.keras.layers.ReLU(),
             
             tf.keras.layers.MaxPool2D(),
             
             tf.keras.layers.Conv2D(filters= 16, kernel_size= (3,3), strides= (2,2), padding= 'same'),
             tf.keras.layers.ReLU(),
             
             tf.keras.layers.MaxPool2D(),
             
             tf.keras.layers.Conv2D(filters= 32, kernel_size= (3,3), strides= (1,1), padding= 'same'),
             tf.keras.layers.ReLU(),
             
             tf.keras.layers.MaxPool2D(),
             
             tf.keras.layers.Flatten(),
             tf.keras.layers.Dense(50),
             tf.keras.layers.ReLU(),
             
             tf.keras.layers.Dense(25),
             tf.keras.layers.ReLU(),
             
             tf.keras.layers.Dense(self.num_actions)
             ])
        return model

    def preprocess_state(self, init_state):
        s1= tf.image.resize(init_state, size= self.state_size)
        s1= tf.expand_dims(s1, 0)
        return s1    
        
    def align_network(self):
        self.target_model1.set_weights(self.qmodel1.get_weights())
        return
    
    def get_action(self, epsilon, s1):
        # exploration
        if np.random.random() < epsilon:
            a1= env.action_space.sample()
        # exploitation
        else:
            qvalues= self.qmodel1(s1)
            a1= np.argmax(qvalues)
        
        if epsilon > self.min_ep:
            redn= epsilon/EPISODES
            epsilon -= redn
        return a1, epsilon
    
    def train_qmodel(self, s1, a1, r1, s2, terminated):
        sample= random.sample(self.replay_memory, BATCH_SIZE)
        for s1, a1, r1, s2, terminated in sample:
            qtarget= self.qmodel1(s1).numpy()
            if terminated:
                qtarget[0, a1]= r1
            else:
                qtarget[0, a1]= r1 + self.gamma*np.max(self.target_model1(s2))
            history= self.qmodel1.fit(s1, qtarget, epochs= 1)
        return history

def episode(agent, timesteps= 100, epsilon= 0.1):
    start= time.time()
    avglosses= []
    for e in range(266, EPISODES+1):
        init_state= env.reset()
        s1= agent.preprocess_state(init_state)
        for timestep in range(1, timesteps+1):
            #if e >= 1900:
            env.render()
            a1, epsilon= agent.get_action(epsilon, s1)
            s2, r1, terminated, info= env.step(a1)
            s2= agent.preprocess_state(s2)
            agent.replay_memory.append((s1, a1, r1, s2, terminated))
            losses= []
            if len(agent.replay_memory) > BATCH_SIZE:
                history= agent.train_qmodel(s1, a1, r1, s2, terminated)
                losses.append(history.history['loss'])
            avgloss= np.sum(losses)/ BATCH_SIZE
            avglosses.append(avgloss)
            s1= s2
            if terminated:
                agent.align_network()
                break
        end= time.time()
        print(f'time taken for training episode-{e} is {end - start}')
    return avglosses


LEARNING_RATE= 0.001
EPISODES= 500
BATCH_SIZE= 16
agent= Agent()
agent.qmodel1.load_weights('/Users/vasudevgupta/Downloads/checkpoints/checkpoints_pacman_dqn_episodes_266/dqn_pacman')
avglosses= episode(agent)

#agent.qmodel1.save_weights('/Users/vasudevgupta/Downloads/checkpoints_pacman_dqn/dqn_pacman')
plt.plot(np.arange(len(avglosses)), avglosses)