import gym
import logging
import os
import imageio

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
from stable_baselines.common.callbacks import EvalCallback

from dialRL.rl_train.environments import DarEnv
from dialRL.models import DQN
from dialRL.utils import get_device
from dialRL.rl_train.callback import MonitorCallback

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)


# policy_model = DQN
env = DarEnv(size=4, target_population=5, driver_population=1)
device = get_device()

print('Env action space :', env.action_space)

monitor = MonitorCallback(eval_env=env,
                          check_freq=1000,
                          log_dir='./data/rl_experiments',
                          n_eval_episodes=5,
                          verbose=1)


model = PPO2(MlpPolicy, env, verbose=0)
model.learn(total_timesteps=10000, callback=monitor)
model.save("Dar_test")

del model # remove to demonstrate saving and loading

model = PPO2.load("Dar_test")
nb_tests = 10

for episode in range(nb_tests):

    # Enjoy trained agent
    rwd = 0
    obs = env.reset()
    env.render()

    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
        rwd += rewards
        if dones :
            break
    print('\t ** \t DONE! \t **')
