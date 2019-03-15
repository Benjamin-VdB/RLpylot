import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from boat_env import BoatEnv

import os
import pickle
os.environ["CUDA_VISIBLE_DEVICES"]="-1" #comment this line if you want to use cuda

# Get the environment and extract the number of actions.
env = BoatEnv()
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

# Next, we build a very simple model.
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()

# simple dqn
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100,
               target_model_update=1e-2, policy=policy)

# dqn with dueling
#dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100,
 #              enable_dueling_network=True, dueling_type='avg', target_model_update=1e-2, policy=policy)

dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.


# Define 'test' for testing an existing network weights or 'train' to train a new one!
mode = 'test'
filename = 'dqn_v1'

if mode == 'train':
    # Train the agent
    hist = dqn.fit(env, nb_steps=200000, visualize=False, verbose=2, nb_max_episode_steps=1000) # 20s episodes
    
    # save history
    with open('_experiments/history_'+ filename + '.pickle', 'wb') as handle:
        pickle.dump(hist.history, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    # After training is done, we save the final weights.
    dqn.save_weights('h5f_files/dqn_{}_weights.h5f'.format(filename), overwrite=True)

    # Finally, evaluate our algorithm for 5 episodes.
    dqn.test(env, nb_episodes=5, visualize=True, nb_max_episode_steps=500)
    
if mode == 'test':
    dqn.load_weights('h5f_files/dqn_{}_weights.h5f'.format(filename))
    dqn.test(env, nb_episodes=10, visualize=True, nb_max_episode_steps=300) # 20 seconds episodes