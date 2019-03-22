import numpy as np
import pickle
import matplotlib.pyplot as plt
from yacht_data import YachtExperiment

filename = 'dqn_500e' # 'ddpg_600kit_rn4_maior2_mem20k_target01_theta3_batch32_adam2'
with open('_experiments/history_' + filename+'.pickle', 'rb') as f:
    hist = pickle.load(f)
f.close()

def _moving_average(a, n=20) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

hist

plt.figure()
plt.title('Reward Evolution')
plt.xlabel('Episodes')
plt.ylabel('Reward')
rw = _moving_average(hist['episode_reward'])
plt.plot(rw)

plt.figure()
plt.title('Survival Evolution')
plt.xlabel('Steps in the episode')
plt.ylabel('Episode')
nsteps = _moving_average(hist['nb_episode_steps'])
plt.plot(nsteps)
plt.show()

# Here you can load and plot you performance test
yachtExp = YachtExperiment()
experiment_name = 'history_dqn_500e.pickle'
yachtExp.load_from_experiment(experiment_name)
yachtExp.plot_obs(iter=-1) # seleciona os episodios manualmente ou coloque -1 para plotar todos
yachtExp.plot_settling_time()
yachtExp.plot_actions(iter=9)