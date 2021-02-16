import numpy as np

import gym
from gym import spaces

import instances
import utils

class DarEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self, size, target_population, driver_population, max_step=100):
        super(DarEnv, self).__init__()

        self.size = size
        self.max_step = max_step
        self.target_population = target_population
        self.driver_population = driver_population
        self.action_space = spaces.Box(low=0,
            high=3,
            shape=(size, size),
            dtype=np.int16)#spaces.Discrete(size**2)
        self.observation_space = spaces.Box(low=0,
            high=3,
            shape=(size, size),
            dtype=np.int16)
        self.max_reward = int(1.5 * self.size)
        self.reward_range = (- self.max_reward, self.max_reward)
        self.current_episode = 0
        self.success_episode = []
        self.cumulative_reward = 0


    def reward(self, distance):
        return self.max_reward - distance #Linear distance
        #int(1.5 * self.size * (1 / distance))


    def reset(self):
        self.instance = instances.PixelInstance(size=self.size,
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
        self.world = utils.instance2world(self.instance.image, self.instance.caracteristics, self.size)
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
        next_pose = utils.heatmap2image_coord(action)
        # next_pose = utils.indice2image_coordonates(action, self.size)

        if self.world[next_pose] ==  -1:
            self.distance = utils.distance(current_pos, next_pose)
            # Update world
            self.del_target(next_pose)
            self.world[next_pose] = self.current_player
            self.world[current_pos] = 0
            # Need to update the vector instace as well ?

        elif self.world[next_pose] > -1 :
            # The world stays unchanged
            self.distance = -1

    def step(self, action):
        self._take_action(action)
        self.current_step += 1

        if self.distance == -1:
            reward = -1#-int(self.max_reward//2)
            done = False
        elif self.distance > 0:
            reward = 1#self.reward(self.distance)
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
        if self.distance == -1 :
            print(f'Player {self.current_player} go lost ....')
        else :
            print(f'Player {self.current_player} aimed right')

        print('World: ')
        print(self.world)
        print('present player: ', self.current_player)
        print('Targets to go: ', self.targets, '\n')


    def render_episode(self, win_or_lose):
        self.success_episode.append(
        'Success' if win_or_lose == 'W' else 'Failure')
        file = open('render/render.txt', 'a')
        file.write(' — — — — — — — — — — — — — — — — — — — — — -\n')
        file.write(f'Episode number {self.current_episode}\n')
        file.write(f'{self.success_episode[-1]} in {self.current_step} steps\n')
        file.close()



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
