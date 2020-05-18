#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 09:39:39 2020

@author: vasudevgupta
"""
import tensorflow as tf
import numpy as np
import pandas as pd
import time
import tqdm
import random
import vizdoom
from vizdoom import *
from collections import deque

game= DoomGame()
path = 'vgbasic.cfg' ## path for configuartion
game.load_config(path)

class initialize(object):
    
    def __init__(self, params):
        self.replay_memory= deque(maxlen= params.maxlen)
        self.qmodel= self.conv_qnetwork()
        self.qmodel.compile(params.optimizer, params.loss)
        self.target_model= self.conv_qnetwork()
    
    def conv_qnetwork(self):
        model= tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters= 8, kernel_size= (3,3), strides= (2,2), 
                                        padding= "same", input_shape= (128, 128, 3), use_bias= True),
                 #tf.keras.layers.BatchNormalization(),
                 tf.keras.layers.ReLU(),
                 
                 tf.keras.layers.Conv2D(filters= 16, kernel_size= (3,3), 
                                        strides= (2,2), padding= 'same', use_bias= True),
                 #tf.keras.layers.BatchNormalization(),
                 tf.keras.layers.ReLU(),
                 
                 tf.keras.layers.Conv2D(filters= 32, kernel_size= (3,3), 
                                        strides= (1,1), padding= 'same', use_bias= True),
                 #tf.keras.layers.BatchNormalization(),
                 tf.keras.layers.ReLU(),
                 
                 tf.keras.layers.Flatten(),
                 tf.keras.layers.Dense(50),
                 tf.keras.layers.ReLU(),
                 #tf.keras.layers.Dropout(0.3),
                 
                 tf.keras.layers.Dense(25),
                 tf.keras.layers.ReLU(),
                 
                 tf.keras.layers.Dense(3)
                 ])
        return model

def getstate_and_preprocess(game):
    state= game.get_state()
    s1= state.screen_buffer
    s1= s1.reshape(240, 320, 3)
    s1= tf.image.resize(s1, size= (128, 128))
    s1= tf.expand_dims(s1, 0)
    return s1
    
def get_action(qvalues, params):
    
    left= [1,0,0]
    right= [0,1,0]
    shoot= [0,0,1]
    actions= [left, right, shoot]
    
    # exploration
    if np.random.random() < params.epsilon:
        a1= random.choice(actions)
    # exploitation
    else:
        a1= np.argmax(qvalues)
        a1= tf.one_hot(a1, depth= len(actions), dtype= 'int32')
        a1= a1.numpy().tolist()
        
    if params.epsilon > params.min_ep:
        redn= params.epsilon/params.episodes
        params.epsilon -= redn
    return a1, params.epsilon

def train_qmodel(game, init, params):
    samples= random.sample(init.replay_memory, params.batch_size)
    for (s1, a1, r1, done, s2) in samples:
        target= init.qmodel(s1, training= True).numpy()
        if done:
            target[0, a1]= r1
        else:
            # double dqn implemetation
            # action value will be measured by Qtarget
            # while action will decided by Qtrain
            t= init.target_model(s2).numpy()
            a2= np.argmax(init.qmodel(s2, training= False).numpy())
            target[0, a1]= r1 + params.gamma*t[0, a2]
        
        histry= init.qmodel.fit(s1, target, verbose= 2, epochs= 1)
        loss= histry.history['loss']
    return histry
        
class params:
    pass

params.num_actions= game.get_available_buttons_size()
params.learning_rate= 0.001
params.loss= tf.keras.losses.mse
params.optimizer= tf.keras.optimizers.Adam(params.learning_rate)
params.timesteps= 100
params.epsilon= 0.6
params.min_ep= 0.1
params.episodes= 2
params.maxlen= 100000
params.batch_size= 32
params.gamma= 0.8

init= initialize(params)
game.init()

for e in tqdm.tqdm(range(1, params.episodes+1)):
    game.new_episode()
    for timestep in range(1, params.timesteps+1):
        s1= getstate_and_preprocess(game)
        qvalues= init.qmodel(s1, training= True)
        a1, params.epsilon= get_action(qvalues, params)
        r1= game.make_action(a1, 2) # same action, we will be taking twice to get better results
        done= game.is_episode_finished()
        if done:
            init.target_model.set_weights(init.qmodel.get_weights())
            break
        s2= getstate_and_preprocess(game)
        init.replay_memory.append((s1, a1, r1, done, s2))
        if len(init.replay_memory) >= params.batch_size:
            loss= train_qmodel(game, init, params)
        
    