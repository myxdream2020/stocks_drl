from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv

import torch.nn as nn
import gym_anytrading.envs as envs
from gym_anytrading.datasets import STOCKS_GOOGL

import stock_utils


StockEnvClass = envs.StocksEnv
# StockEnvClass = stock_utils.MyStocksEnv

# train env 
window_size = 6
env_factory = lambda: StockEnvClass(STOCKS_GOOGL, window_size=window_size, frame_bound=(window_size, len(STOCKS_GOOGL)))
env = DummyVecEnv([env_factory])


# test env 
test_data = STOCKS_GOOGL.tail(150)
test_env = StockEnvClass(test_data, window_size=window_size, frame_bound=(window_size, len(test_data)))

# tain model 
test_callback = stock_utils.TestModelCallback(eval_env=test_env, eval_freq=5000)
policy = "MlpPolicy"
model = A2C(policy, env, verbose=1, learning_rate=0.0001)
model.learn(total_timesteps=300_000, callback=test_callback, log_interval=5000)

# evaluate 
print("evaluate model: ")
stock_utils.evaluate_model2(model, test_env, True)

input("finished! press enter to exit...")