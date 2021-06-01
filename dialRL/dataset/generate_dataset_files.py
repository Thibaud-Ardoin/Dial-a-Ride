from dialRL.rl_train.environments import DarEnv, DarPixelEnv, DarSeqEnv
from dialRL.utils import get_device, trans25_coord2int
from dialRL.rl_train.callback import MonitorCallback
from dialRL.strategies import NNStrategy

params = {
    'number'
}


environment = DarSeqEnv(size=size,
                        target_population=target_population,
                        driver_population=driver_population,
                        reward_function=self.rwd_fun,
                        time_end=time_end,
                        max_step=max_step,
                        timeless=timeless,
                        dataset=self.data,
                        test_env=test_env)



dir = './data/rl_experiments/strategies/' + \
   self.__class__.__name__
os.makedirs(dir, exist_ok=True)
cumulative_reward = 0
observation = self.env.reset()
images = []
last_time = 0
self.env.render()
for t in range(self.max_step):
action = self.action_choice(observation)
observation, reward, done, info = self.env.step(action)
cumulative_reward += reward
print('Cumulative reward : ', cumulative_reward, ' (', reward, ')')
self.env.render()

# If time updataed, save image
if self.env.time_step > last_time :
    last_time = self.env.time_step
    images.append(self.save_image(dir, t, self.env.get_image_representation()))

if done:
    print("\n ** \t Episode finished after {} time steps".format(t+1))
    break

self.save_video(dir, images)
self.env.close()
