import gym
import gym_abr
import numpy as np
from RL_brain import DeepQNetwork
import os

S_INFO = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6

env = gym.make('ABR-v0')

RL = DeepQNetwork(n_actions=A_DIM, n_features=S_INFO*S_LEN, learning_rate=1e-4, e_greedy=0.99,
                  replace_target_iter=100, memory_size=3000,
                  e_greedy_increment=1e-6,)

_file = open('test.csv', 'w')
step = 0
episode = 0
while True:
#for episode in range(3000):
    # initial observation
    ep_r = 0.
    fetch = 0.
    observation = env.reset()
    observation = np.reshape(observation, (S_INFO*S_LEN))
    while True:
        action = RL.choose_action(observation)
        observation_, reward, done, _ = env.step(action)
        observation_ = np.reshape(observation_, (S_INFO*S_LEN))
        RL.store_transition(observation, action, reward, observation_)
        ep_r += reward
        if (step > 200) and (step % 5 == 0):
            RL.learn()
        # swap observation
        observation = observation_
        # break while loop when end of this episode
        fetch += 1
        if done:
            break
        step += 1
    ep_r /= fetch
    if episode % 100 == 0:
        print('Epi: ', episode,
            ', Ep_r: ', round(ep_r, 4),
            ', Epsilon: ', round(RL.epsilon, 2))
    _file.write(str(ep_r) + '\n')
    _file.flush()
    episode += 1

_file.close()
