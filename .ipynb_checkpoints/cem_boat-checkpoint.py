import numpy as np
import gym


from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.cem import CEMAgent
from rl.memory import EpisodeParameterMemory
from boat_env import BoatEnv

import os
import pickle
os.environ["CUDA_VISIBLE_DEVICES"]="-1" #comment this line if you want to use cuda

# Get the environment and extract the number of actions.
env = BoatEnv(type='discrete')
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n
obs_dim = env.observation_space.shape[0]

# Option 1 : Simple model
# model = Sequential()
# model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
# model.add(Dense(nb_actions))
# model.add(Activation('softmax'))

# Option 2: deep network
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('softmax'))

print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = EpisodeParameterMemory(limit=1000, window_length=1)

cem = CEMAgent(model=model, nb_actions=nb_actions, memory=memory,
               batch_size=50, nb_steps_warmup=2000, train_interval=50, elite_frac=0.05)
cem.compile()


# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.


# Define 'test' for testing an existing network weights or 'train' to train a new one!
mode = 'train'
filename = 'cem_discrete10'

if mode == 'train':
    # Train the agent
    tensorb = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=100, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='batch')
    
    tb = TensorBoard(log_dir='./logs/log_{}'.format(filename))
    
    hist = cem.fit(env, nb_steps=200000, visualize=False, verbose=2, nb_max_episode_steps=500, callbacks=[tb]) # 20s episodes
    
    # print history
    print("history contents : ", hist.history.keys()) # episode_reward, nb_episode_steps, nb_steps
    # summarize history for accuracy
    import matplotlib.pyplot as plt
    plt.plot(hist.history['episode_reward'])
    plt.plot(hist.history['nb_episode_steps'])
    plt.title('learning')
    plt.xlabel('episode')
    plt.legend(['episode_reward', 'nb_episode_steps'], loc='upper left')
    plt.show()
    
    # save history
    with open('_experiments/history_'+ filename + '.pickle', 'wb') as handle:
        pickle.dump(hist.history, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    # After training is done, we save the final weights.
    cem.save_weights('h5f_files/dqn_{}_weights.h5f'.format(filename), overwrite=True)

    # Finally, evaluate our algorithm for 5 episodes.
    cem.test(env, nb_episodes=5, visualize=True, nb_max_episode_steps=500)
    
if mode == 'test':
    cem.load_weights('h5f_files/dqn_{}_weights.h5f'.format(filename))
    cem.test(env, nb_episodes=10, visualize=True, nb_max_episode_steps=400) # 40 seconds episodes