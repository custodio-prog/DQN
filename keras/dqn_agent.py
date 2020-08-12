# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 18:02:41 2019

@author: custodio
"""

import numpy as np 
import random 
import time 
import os 
from random import randint 
import pandas as pd 
import logging 
import time 
import itertools
import pickle 
from random import  sample
from datetime import datetime
import h5py
from numba import jit
import dask.array as da
import dask.delayed as delay
from numba import jitclass          
from numba import int32, float32
from joblib import Parallel, delayed
import multiprocessing
import random
import gym
import numpy as np
from collections import deque
# from keras.models import Sequential
# from keras.layers import Dense
from keras import backend as K
from tensorflow.keras import layers, Sequential
import tensorflow as tf
from math import sqrt
from statistics import mean
VERYLARGENUMBER = np.inf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    format='[%(asctime)s] %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

class DQNAgent(object):    
    def __init__(self, settings, state_size, action_size):  # Runs when instance of the class is created
        
        self.settings = settings
        self.action_size = action_size
        self.state_size = state_size
        self.npts = round(24/(self.settings.time_step/3600)) # number of episodes in q-learning
        d = datetime.now()
        self.index = d.hour+d.minute+d.second
        
        #New
        # self.loss_function = tf.keras.losses.MeanSquaredError()
        self.loss_function = tf.keras.losses.LogCosh()
        # self.loss_function = tf.keras.losses.Huber()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=settings.learning_rate)
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        self.memory = deque(maxlen=self.settings.memory_capacity)
  
    def _build_model(self):

        def keras_opt(): # HINT *KERAS is used here 
            # Neural Net for Deep-Q learning Model
            model = Sequential()
            num_hidden_neurons = round(sqrt(self.state_size*self.action_size))
            logging.info(f'Number of hidden layers is {num_hidden_neurons}')
            model.add(layers.Dense(10*num_hidden_neurons, 
            input_dim=self.state_size, activation = self.settings.afun, 
            kernel_initializer='glorot_uniform', 
            kernel_regularizer=regularizers.l2(l=self.settings.l)))
            model.add(layers.Dense(5*num_hidden_neurons, 
            activation = self.settings.afun, 
            kernel_regularizer=regularizers.l2(l=self.settings.l)))
            model.add(layers.Dense(2*num_hidden_neurons, 
            activation = self.settings.afun, 
            kernel_regularizer=regularizers.l2(l=self.settings.l)))
            # model.add(layers.Dense(2*num_hidden_neurons, activation='relu'))
            # model.add(layers.Dense(1*num_hidden_neurons, activation='relu'))
            model.add(layers.Dense(self.action_size, activation='linear'))
            model.compile(loss=self.loss_function, optimizer=self.optimizer)
            return model
        
        def tf_opt():

            num_hidden_neurons = round(sqrt(self.state_size*self.action_size))
            logging.info(f'Number of hidden layers is {num_hidden_neurons}')
            # Network defined by the Deepmind paper
            inputs = tf.keras.layers.Input(shape=(self.state_size,))
            # Dense layers
            layer1 = tf.keras.layers.Dense(4*num_hidden_neurons, activation='relu')(inputs)
            layer2 = tf.keras.layers.Dense(4*num_hidden_neurons, activation='relu')(layer1)
            action = tf.keras.layers.Dense(self.action_size, activation='softmax')(layer2)
            model = tf.keras.Model(inputs=inputs, outputs=action)  
            model.compile(loss=self.loss_function, optimizer=self.optimizer)
            return model
        
        model = tf_opt()
        return model       

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def memorize(self, state, action, reward, next_state, noviol):
        self.memory.append((state, action, reward, next_state, noviol))

    def act(self, state):
        if np.random.rand() <= self.settings.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self):
        def keras_opt(states, actions, rewards, next_states, noviols): #*chosen one
            targets = self.model.predict(states)
            future_rewards = self.target_model.predict(next_states, workers=4, use_multiprocessing=True)
            expected_returns= rewards + self.settings.gamma*np.max(future_rewards, axis=1) 
            actual_values = np.where(noviols, rewards, expected_returns)

            for t,a,av in zip(targets,actions,actual_values):
                t[a] = av

            result = self.model.fit(states, targets, epochs=1, batch_size=self.settings.batch_size, verbose=0)
            loss = result.history['loss'][0]

        def tf_opt(states, actions, rewards, next_states, noviols):
            # Build the updated Q-values for the sampled future states
            # Use the target model for stability
            future_rewards = self.target_model.predict(next_states)
            # Q value = reward + discount factor * expected future reward
            updated_q_values = rewards + self.settings.gamma * tf.reduce_max(
                future_rewards, axis=1
            )
            # If final frame set the last value to -1
            updated_q_values = updated_q_values * (1 - noviols) + noviols*rewards

            # Create a mask so we only calculate loss on the updated Q-values
            masks = tf.one_hot(actions, self.action_size)

            with tf.GradientTape() as tape:
                # Train the model on the states and updated Q-values
                q_values = self.model(states)
                # Apply the masks to the Q-values to get the Q-value for action taken
                q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                # Calculate loss between new Q-value and old Q-value
                loss = self.loss_function(updated_q_values, q_action)

            # Backpropagation
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            return float(loss)

        minibatch = random.sample(self.memory, self.settings.batch_size)
        states = np.asarray([sample[0][0] for sample in minibatch])
        actions = np.asarray([sample[1] for sample in minibatch])
        rewards = np.asarray([sample[2] for sample in minibatch])
        next_states = np.asarray([sample[3][0] for sample in minibatch])
        noviols = np.asarray([sample[4] for sample in minibatch])
        loss = tf_opt(states, actions, rewards, next_states, noviols)
        self.settings.epsilon *= self.settings.epsilon_decay
        self.settings.epsilon = max(self.settings.epsilon, self.settings.epsilon_min)
        return loss 
       
    def load(self, name):
        self.model.load_weights(f'{self.settings.outp_folder}/weights_{name}')
        self.target_model.load_weights(f'{self.settings.outp_folder}/weights_{name}')

    def save(self, name, ep_rh,loss_h):
        with open(f'{self.settings.outp_folder}/eprh_{name}.pkl', 'wb') as f:
                pickle.dump(ep_rh, f, pickle.HIGHEST_PROTOCOL)

        with open(f'{self.settings.outp_folder}/lossh_{name}.pkl', 'wb') as f:
                pickle.dump(loss_h, f, pickle.HIGHEST_PROTOCOL)
        
        self.model.save_weights(f'{self.settings.outp_folder}/weights_{name}')   