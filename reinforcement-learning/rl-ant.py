# %%
import os
import gymnasium as gym
#https://github.com/DLR-RM/stable-baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

env_name = "Ant-v4"
log_path = os.path.join('Training', 'Ant', 'Logs')
PPO_path = os.path.join('Training', 'Ant', 'Models', 'PPO_Model_Cp')

env = gym.make(env_name)
vec_env = DummyVecEnv([lambda: env])
