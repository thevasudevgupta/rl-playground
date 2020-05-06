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

env= gym.make('MsPacman-v0')

def conv_qnetwork(num_sctions):
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
             
             tf.keras.layers.Dense(num_actions)
             ])
        return model

def preprocess_state(init_state, init_state_size):
    s1= tf.image.resize(init_state, size= init_state_size)
    s1= tf.expand_dims(s1, 0)
    return s1    
    
def align_network(target_model, qmodel):
    target_model.set_weights(qmodel.get_weights())
    return

def get_action(epsilon, min_ep, s1):
    # exploration
    if np.random.random() < epsilon:
        a1= env.action_space.sample()
    # exploitation
    else:
        qvalues= qmodel(s1)
        a1= np.argmax(qvalues)
    
    if epsilon > min_ep:
        redn= epsilon/EPISODES
        epsilon -= redn
    return a1, epsilon

def train_qmodel(s1, a1, r1, s2, terminated, target_model, qmodel):
    sample= random.sample(replay_memory, BATCH_SIZE)
    for s1, a1, r1, s2, terminated in sample:
        qtarget= qmodel(s1).numpy()
        if terminated:
            qtarget[0, a1]= r1
        else:
            qtarget[0, a1]= r1 + GAMMA*np.max(target_model(s2))
        history= qmodel.fit(s1, qtarget, epochs= 1)
    return history

len_rpm= 10000
replay_memory= deque(maxlen= len_rpm)
GAMMA= 0.9
epsilon= 0.5
min_ep= 0.1
LEARNING_RATE= 0.001
BATCH_SIZE= 3
init_state_size= (124, 124)
timesteps= 10
EPISODES= 2
num_actions= env.action_space.n

optimizer= tf.keras.optimizers.Adam(LEARNING_RATE)
qmodel= conv_qnetwork(num_actions)
qmodel.compile(optimizer, loss= 'mse')

target_model= conv_qnetwork(num_actions)


for episode in range(1, EPISODES+1):
    init_state= env.reset()
    s1= preprocess_state(init_state, init_state_size)
    
    for timestep in range(1, timesteps+1):
        a1, epsilon= get_action(epsilon, min_ep, s1)
        s2, r1, terminated, info= env.step(a1)
        s2= preprocess_state(s2, init_state_size)
        replay_memory.append((s1, a1, r1, s2, terminated))
        
        if len(replay_memory) > BATCH_SIZE:
            history= train_qmodel(s1, a1, r1, s2, terminated, target_model, qmodel)
        
        s1= s2
        if terminated:
            align_network(target_model, qmodel)
            break
