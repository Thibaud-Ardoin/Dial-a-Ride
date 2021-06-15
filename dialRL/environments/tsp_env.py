import numpy as np

import gym
from gym import spaces

from dialRL.environments import PixelInstance
from dialRL.utils import instance2world, indice2image_coordonates, distance

class TspEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self, size, target_population, driver_population, max_step=10):
        super(DarEnv, self).__init__()

        self.size = size
        self.max_step = max_step
        self.target_population = target_population
        self.driver_population = driver_population
        self.action_space = spaces.Discrete(size**2)
        # spaces.Box(low=-1,
        #     high=1,
        #     shape=(size, size),
        #     dtype=np.int16)#spaces.Discrete(size**2)
        self.observation_space = spaces.Box(low=-1,
            high=3,
            shape=(size, size),
            dtype=np.int16)
        self.max_reward = int(1.5 * self.size)
        self.reward_range = (- self.max_reward, self.max_reward)
        self.current_episode = 0
        self.success_episode = []
        self.cumulative_reward = 0
        self.last_aim = None


    def reset(self):
        self.instance = PixelInstance(size=self.size,
                                                population=self.target_population,
                                                drivers=self.driver_population,
                                                moving_car=True,
                                                transformer=True,
                                                verbose=False)
        self.instance.random_generation()
        # print('* Reset - Instance image : ', self.instance.image)
        self.targets = self.instance.points.copy()
        self.drivers = self.instance.centers.copy()

        self.current_player = 1
        # distance is -1 if wrong aiming. 0 if there is no start of game yet and x if aimed corectly
        self.distance = 0
        self.current_step = 0
        self.cumulative_reward = 0
        self.world = instance2world(self.instance.image, self.instance.caracteristics, self.size)
        self.last_aim = None
        return self._next_observation()


    def del_target(self, position):
        filter_fun = lambda x : x[0] == position[0] and x[1] == position[1]
        indices = [i for i in range(len(self.targets)) if filter_fun(self.targets[i])]
        for indice in indices:
            del self.targets[indice]


    def _next_observation(self):
        obs = self.world
        # addition_info = np.zeros(self.size)
        # addition_info[0] = self.current_player
        # obs = np.append(obs, np.array([addition_info]), axis=0)
        return obs


    def _take_action(self, action):
        """ Action: destination point as an indice of the map vactor. (Ex: 1548 over 2500)
        """

        current_pos = np.where(self.world == self.current_player)
        # indice = np.argmax(action)
        next_pose = indice2image_coordonates(action, self.size)
        # next_pose = utils.indice2image_coordonates(action, self.size)
        self.last_aim = next_pose

        if self.world[next_pose] ==  -1:
            self.distance = distance(current_pos, next_pose)
            # Update world
            self.del_target(next_pose)
            self.world[next_pose] = self.current_player
            self.world[current_pos] = 0
            # Need to update the vector instace as well ?

        elif self.world[next_pose] > -1 :
            # The world stays unchanged
            self.distance = -1


    def reward(self, distance):
        return self.max_reward - distance #Linear distance
        #int(1.5 * self.size * (1 / distance))


    def step(self, action):
        # Curently the action is a heatmap ..
        # NOO now it's an indice in the list of action n*n

        self._take_action(action)
        self.current_step += 1

        if self.distance == -1:
            reward = -1 #-int(self.max_reward//2)
            done = False
        elif self.distance > 0:
            reward = self.reward(self.distance)
            done = False
            # update drivers turn
            self.current_player = ((self.current_player + 1 - 1) % (self.driver_population) ) + 1
        # End of simulation

        self.cumulative_reward += reward

        if not len(self.targets) :
            done = True
        if self.current_step >= self.max_step:
            done = True

        if done:
            self.current_episode += 1
            # print('End of episode, total reward :', self.cumulative_reward)

        obs = self._next_observation()

        # Last element is info (dict)
        return obs, reward, done, {}


    def render(self):
        print('\n--------------------- [Step', self.current_step, ']')
        print('World: ')
        print(self.world)

        if self.distance == -1 :
            print(f'Player {self.current_player} go lost ....')
        else :
            print(f'Player {self.current_player} aimed right')

        print('present player: ', self.current_player)
        print('Last aimed to : ', self.last_aim)
        print('Targets to go: ', self.targets)
        print('Cumulative reward : ', self.cumulative_reward)
        print('---------------------\n')


if __name__ == '__main__':

    env = DarEnv(size=8, target_population=5, driver_population=2)
    cumulative_reward = 0
    observation = env.reset()
    for t in range(100):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        cumulative_reward += reward
        print('Cumulative reward : ', cumulative_reward, ' (', reward, ')')
        if done:
            print("\n ** \t Episode finished after {} timesteps".format(t+1))
            env.render()
            break
    env.close()
