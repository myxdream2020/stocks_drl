import gym_anytrading.envs as envs
from gym_anytrading.envs import Actions 
from stable_baselines3.common.callbacks import BaseCallback
from matplotlib import pyplot as plt 
import numpy as np

class MyStocksEnv(envs.StocksEnv):

    def _calculate_reward(self, action):
        step_reward = 0

        if self._current_tick >= (len(self.prices) - 1):
            return step_reward

        current_price = self.prices[self._current_tick]
        prev_day_price = self.prices[self._current_tick-1]
        price_diff = current_price - prev_day_price

        if price_diff > 0:
            step_reward += 1 if action == Actions.Buy.value else 0
        else:
            step_reward += 1 if action == Actions.Sell.value else 0
        
        return step_reward


def evaluate_model2(model, test_env, debug=False):
    
    obs, info = test_env.reset()
    episode_reward = 0

    step = 0
    while True:
        step += 1
        action, _states = model.predict(obs, deterministic=True)

        obs, rewards, terminated, truncated, info = test_env.step(action)
        episode_reward += rewards

        if debug:
            print(f"{step}\t{action}\t{rewards}")

        done = terminated or truncated
        if done:
            # print("info", info)
            break
    
    if debug:
        plt.figure(figsize=(15,6))
        plt.cla()
        test_env.render_all()
        plt.show()

    # return episode_reward
    return info['total_reward'], info['total_profit']



class TestModelCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=10000, verbose=1):
        super(TestModelCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq == 0:
            mean_reward, std_reward, mean_profit, std_profit = self.evaluate_model()
            self.logger.record("eval/mean_reward", mean_reward)
            self.logger.record("eval/std_reward", std_reward)
            self.logger.record("eval/mean_profit", mean_profit)
            self.logger.record("eval/std_profit", std_profit)
            self.logger.dump(step=self.num_timesteps)

        return True

    def evaluate_model(self):
        num_episodes = 1
        rewards = []
        profits = []

        for _ in range(num_episodes):
            info = evaluate_model2(self.model, self.eval_env)
            rewards.append(info[0])
            profits.append(info[1])

        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)

        mean_profit = np.mean(profits)
        std_profit = np.std(profits)

        return mean_reward, std_reward, mean_profit, std_profit
