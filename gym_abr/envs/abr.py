# add queuing delay into halo
import os
import numpy as np
import env
import load_trace
import gym
from gym import spaces
from gym.utils import seeding

S_INFO = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
TRAIN_SEQ_LEN = 100  # take as a train batch
MODEL_SAVE_INTERVAL = 100
VIDEO_BIT_RATE = [300,750,1200,1850,2850,4300]  # Kbps
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000


class ABREnv(gym.Env):

    def __init__(self, random_seed=RANDOM_SEED):
        np.random.seed(RANDOM_SEED)
        self.action_space = spaces.Discrete(A_DIM)
        self.observation_space = spaces.Box(
            0, 10.0, [S_INFO, S_LEN], dtype=np.float32)
        all_cooked_time, all_cooked_bw, _ = load_trace.load_trace()
        self.net_env = env.Environment(all_cooked_time=all_cooked_time,
                            all_cooked_bw=all_cooked_bw)
        
        self.last_bit_rate = DEFAULT_QUALITY
        self.state = np.zeros((S_INFO, S_LEN))
        self.reset()

    def reset(self):
        self.time_stamp = 0
        self.last_bit_rate = DEFAULT_QUALITY
        self.state = np.zeros((S_INFO, S_LEN))
        return self.state

    def step(self, action):
        bit_rate = action
        # the action is from the last decision
        # this is to make the framework similar to the real
        delay, sleep_time, buffer_size, rebuf, \
            video_chunk_size, next_video_chunk_sizes, \
            end_of_video, video_chunk_remain = \
            self.net_env.get_video_chunk(bit_rate)

        self.time_stamp += delay  # in ms
        self.time_stamp += sleep_time  # in ms

        # reward is video quality - rebuffer penalty - smooth penalty
        reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
            - REBUF_PENALTY * rebuf \
            - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                      VIDEO_BIT_RATE[self.last_bit_rate]) / M_IN_K

        self.last_bit_rate = bit_rate
        state = np.roll(self.state, -1, axis=1)

        # this should be S_INFO number of terms
        state[0, -1] = VIDEO_BIT_RATE[bit_rate] / \
            float(np.max(VIDEO_BIT_RATE))  # last quality
        state[1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
        state[2, -1] = float(video_chunk_size) / \
            float(delay) / M_IN_K  # kilo byte / ms
        state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
        state[4, :A_DIM] = np.array(
            next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
        state[5, -1] = np.minimum(video_chunk_remain,
                                  CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)

        self.state = state
        #observation, reward, done, info = env.step(action)
        return state, reward, end_of_video, {}
