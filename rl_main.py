import numpy as np
import gym

from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

#from rl.agents.dqn import DQNAgent
from rl.agents import DQNAgent, SARSAAgent
from rl.policy import BoltzmannQPolicy, GreedyQPolicy
from rl.memory import SequentialMemory
from boat_env import BoatEnv

import os
import pickle

import argparse

parser = argparse.ArgumentParser(description='RLpylot training and testing options')

"----------------------------- General options -----------------------------"
parser.add_argument('--learnmode', default='train', type=str, help='train/test/real')

opt = parser.parse_args()

# Prog arguments
args = opt

if __name__ == "__main__":
    
    # Boat environment
    env = BoatEnv(type='discrete', mode='simulation')
    np.random.seed(123)
    env.seed(123)
    nb_actions = env.action_space.n
    
    # Model definition
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape)) 
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    print(model.summary())
    
    # Agent Config
    """
    Tests
    dqn1 : 3 layers 128-16
    dqn2 : 4 layers 256-16
    dqn3 : 5 layers 512-16
    dqn4 : 4 layers 256-16 + dueling
    dqn5 : 4 layers 256-16 + double dqn
    dqn6 : 4 layers 256-16 + double dqn + dueling
    """

    memory = SequentialMemory(limit=50000, window_length=1)
    policy = BoltzmannQPolicy()
    # policy = GreedyQPolicy()
    # policy = None # EPSgreedy in training / Greedy in test
    
    agent = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=1000, enable_double_dqn=True,
                     enable_dueling_network=True,
                     target_model_update=1e-2, policy=policy)
    
    # dqn with dueling
    # dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100,    
    #                enable_dueling_network=True, dueling_type='avg', target_model_update=1e-2, policy=policy)
    agent.compile(Adam(lr=1e-3), metrics=['mae'])
    """
    
    # SARSA does not require a memory.
    policy = BoltzmannQPolicy()
    agent = SARSAAgent(model=model, nb_actions=nb_actions, nb_steps_warmup=100, policy=policy)
    agent.compile(Adam(lr=1e-3), metrics=['mae'])
    """#
    
    # Training and testing modes
    mode = args.learnmode
    filename = 'dqn_6' #dqn_discrete5_4lay_epsgreedpol_rwd0.01and_neg0.1_targetonly_60deg'
    
    if mode == 'train':
        # Train the agent
        
        tb = TensorBoard(log_dir='./logs/log_{}'.format(filename))
        
        hist = agent.fit(env, nb_steps=100000, visualize=True, verbose=2, nb_max_episode_steps=500, callbacks=[tb]) # 20s episodes
        
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
        agent.save_weights('h5f_files/dqn_{}_weights.h5f'.format(filename), overwrite=True)
    
        # Finally, evaluate our algorithm for 5 episodes.
        agent.test(env, nb_episodes=5, visualize=True, nb_max_episode_steps=500)
        
    if mode == 'test':
        agent.load_weights('h5f_files/dqn_{}_weights.h5f'.format(filename))
        agent.test(env, nb_episodes=10, visualize=True, nb_max_episode_steps=400) # 40 seconds episodes
        
        
    if mode == 'real':
        
        # set the heading target
        env.target = 0.
        
        agent.load_weights('h5f_files/dqn_{}_weights.h5f'.format(filename))
        agent.test(env, nb_episodes=10, visualize=True, nb_max_episode_steps=400) # 40 seconds episodes