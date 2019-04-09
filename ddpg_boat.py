import numpy as np
import gym


from keras.callbacks import TensorBoard
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from boat_env import BoatEnv

import os
import pickle
os.environ["CUDA_VISIBLE_DEVICES"]="-1" #comment this line if you want to use cuda


# Get the environment and extract the number of actions.
env = BoatEnv(type='continuous')
np.random.seed(123)
env.seed(123)
assert len(env.action_space.shape) == 1
nb_actions = env.action_space.shape[0]
#nb_actions = env.action_space.n
print(nb_actions)

# Next, we build a very simple model.
actor = Sequential()
actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
actor.add(Dense(64))
actor.add(Activation('relu'))
actor.add(Dense(32))
actor.add(Activation('relu'))
actor.add(Dense(16))
actor.add(Activation('relu'))
actor.add(Dense(nb_actions))
actor.add(Activation('linear'))
print(actor.summary())

action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
flattened_observation = Flatten()(observation_input)
x = Concatenate()([action_input, flattened_observation])
x = Dense(128)(x)
x = Activation('relu')(x)
x = Dense(64)(x)
x = Activation('relu')(x)
x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(1)(x)
x = Activation('linear')(x)
critic = Model(inputs=[action_input, observation_input], outputs=x)
print(critic.summary())

# DDPG Agent for continous action space
memory = SequentialMemory(limit=100000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3)

agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                  random_process=random_process, gamma=.5, target_model_update=1e-3)
agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])



# Training or Testing
mode = 'train'
filename = 'ddpg_continuous_gamma0.5'

if mode == 'train':
    # Train the agent    
    tb = TensorBoard(log_dir='./logs/log_{}'.format(filename))
    
    hist = agent.fit(env, nb_steps=50000, visualize=False, verbose=1, nb_max_episode_steps=500, callbacks=[tb])
    
      
    # save history
    with open('_experiments/history_'+ filename + '.pickle', 'wb') as handle:
        pickle.dump(hist.history, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    # After training is done, we save the final weights.
    agent.save_weights('h5f_files/dqn_{}_weights.h5f'.format(filename), overwrite=True)

    # Finally, evaluate our algorithm for 5 episodes.
    agent.test(env, nb_episodes=5, visualize=True, nb_max_episode_steps=500)
    
if mode == 'test':
    agent.load_weights('h5f_files/dqn_{}_weights.h5f'.format(filename))
    agent.test(env, nb_episodes=10, visualize=True, nb_max_episode_steps=300) # 20 seconds episodes