# %%
import os
import gymnasium as gym
#https://github.com/DLR-RM/stable-baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

env_name = "CartPole-v1"
log_path = os.path.join('Training', 'Logs')
PPO_path = os.path.join('Training', 'Models', 'PPO_Model_Cp')

# %% defining and training model
env = gym.make(env_name)
env = DummyVecEnv([lambda: env])
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)
model.learn(total_timesteps=20000)
model.save(PPO_path)

env.close()
del model


# %%
env = gym.make(env_name, render_mode='human')
env = DummyVecEnv([lambda: env])
model = PPO.load(PPO_path, env=env)

# %% evaluating model
evaluate_policy(model=model, env=env, n_eval_episodes=10, render=False)

# %% evaluating model by own loop
episodes = 5
for episode in range(1, episodes+1):
  obs = env.reset()
  done = False
  score = 0

  while not done:
    env.render()
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    score += reward
  print(f'Episode: {episode}, Score: {score}')
env.close()
del model

# %% callback to training stage
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

env = gym.make(env_name)
env = DummyVecEnv([lambda: env])

stop_callback = StopTrainingOnRewardThreshold(reward_threshold=500, verbose=1)
eval_callback = EvalCallback(env,
                             callback_on_new_best=stop_callback,
                             eval_freq=1000,
                             best_model_save_path=best_model_path,
                             verbose=1
                             )

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)
model.learn(total_timesteps=20000, callback=eval_callback)

# %%
env = gym.make(env_name, render_mode='human')
model.set_env(env)

evaluate_policy(model=model, env=env, n_eval_episodes=2, render=False)

# %% changing policies f.e. custom layers
net_arch = [dict(pi=[128,128,128,128], vf=[128,128,128,128])]
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path, policy_kwargs={'net_arch': net_arch})
model.learn(total_timesteps=20000, callback=eval_callback)

# %%
evaluate_policy(model=model, env=env, n_eval_episodes=10, render=True)

# %%