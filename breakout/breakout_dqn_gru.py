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

env= gym.make('Breakout-v0')

def preprocess_state(init_state):
    s1= tf.image.resize(init_state, size= (299, 299))
    s1= tf.keras.applications.inception_resnet_v2.preprocess_input(s1)
    s1= tf.expand_dims(s1, 0)
    return s1    
    
def get_action(qvalues, epsilon, min_ep= 0.1):
    # exploration
    if np.random.random() < epsilon:
        a1= env.action_space.sample()
    # exploitation
    else:
        a1= np.argmax(qvalues)
        
    if epsilon > min_ep:
        redn= epsilon/EPISODES
        epsilon -= redn
    return a1, epsilon

###############################################################################
"""
Model Architecture
"""
premodel= tf.keras.applications.InceptionResNetV2(weights= 'imagenet', include_top= True)
input_layer= premodel.input
output_layer= premodel.output
cnn_model= tf.keras.Model(inputs= input_layer, outputs= output_layer)

class gru_qnetwork(tf.keras.Model):
    
    def __init__(self):
        super(gru_qnetwork, self).__init__()
        self.dense= tf.keras.layers.Dense(25, input_shape= (1000,))
        self.reshape= tf.keras.layers.Reshape((1, 25))
        self.l1= tf.keras.layers.GRU(units= 10, return_state= True, input_shape= (1, 25))
        self.l2= tf.keras.layers.Dense(4)
        
    def call(self, x, gru_state):
        x= self.dense(x)
        x= self.reshape(x)
        x, state= self.l1(x, initial_state= gru_state)
        x= self.l2(x)
        return x, state
## reason for using reshape layer ??
###############################################################################
"""
Model defining and Hyperparameters
"""
gru_train= gru_qnetwork()
gru_target= gru_qnetwork()
seqlen= 6
episode_loss= []
optimizer= tf.keras.optimizers.Adam(learning_rate= 0.01)
timesteps= 5
gamma= 0.9
epsilon= 0.2
EPISODES= 2

###############################################################################
""" 
Model Training 
"""
for e in range(1, EPISODES+1):
    print(f'||||||||||||||||||EPISODE-{e} begins|||||||||||||||||||')
    s1= env.reset()
    x= deque(maxlen= seqlen)
    s1= preprocess_state(s1)
    start= time.time()
    losses= []
    for j in range(1, timesteps+1):
        gru_state= tf.expand_dims(tf.zeros(10), 0)
        a= time.time()
        with tf.GradientTape() as gtape:
            for i in range(seqlen):
                x.append(cnn_model(s1, training= False))
                qval, gru_state= gru_train(x[i], gru_state)
                a1, epsilon= get_action(qval, epsilon)
                s2, r1, done, _= env.step(a1)
                if done:
                    break
                s2= preprocess_state(s2)
                s1= s2
            qtarget, _= gru_train(x[seqlen-1], gru_state, training= False)
            qtarget= qtarget.numpy()
            if done:
                qtarget[0, a1]= r1
                print(f'time taken for training timestep-{j} is {time.time() - a}')
                break
            else:
                xfinal= cnn_model(s2, training= False)
                qt, _= gru_target(xfinal, gru_state, training= False)
                qtarget[0, a1]= r1 + gamma*tf.math.reduce_max(qt)
            loss= tf.keras.losses.mse(qtarget, qval)
        grads= gtape.gradient(loss, gru_train.trainable_variables)
        optimizer.apply_gradients(zip(grads, gru_train.trainable_variables))
        losses.append(loss.numpy())
        print(f'TIMESTEP-{j} LOSS: {loss}')
        b= time.time()
        if j>= 50:
            gru_target.set_weights(gru_train.get_weights())
        print(f'time taken for training timestep-{j} is {b - a}')
    episode_loss.append(np.mean(loss))
    gru_train.save_weights('/Users/vasudevgupta/Downloads/checkpoints/checkpoints_breakout_dqn/dqn_breakout')
    print('----------------------MODEL SAVED----------------------')
    print(f'EPISODE-{e} LOSS: {np.mean(loss)}')
    end= time.time()
    print(f'total time taken for training episode-{e} is {end - start}')

###############################################################################
# agent.qmodel1.load_weights('/Users/vasudevgupta/Downloads/checkpoints/checkpoints_breakout_dqn/dqn_breakout')
gru_train.save_weights('/Users/vasudevgupta/Downloads/checkpoints/checkpoints_breakout_dqn/dqn_breakout')
plt.plot(np.arange(len(episode_loss)), episode_loss)

###############################################################################
"""
Rendering environment
"""
s1= env.reset()
s1= preprocess_state(s1)
timesteps= 1000

for timestep in range(1, timesteps+1):
    env.render()
    m= cnn_model(s1, training= False)
    qval, gru_state= gru_train(m, gru_state, training= False)
    a1, epsilon= get_action(qval, epsilon, min_ep= epsilon)
    s2, r2, done, _= env.step(a1)
    s2= preprocess_state(s2)
    s1= s2
    if done:
        break
    #time.sleep(0.00001)
###############################################################################