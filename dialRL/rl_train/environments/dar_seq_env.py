import numpy as np

import gym
from gym import spaces
import drawSvg as draw
import tempfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dialRL.dataset import tabu_parse_info
from dialRL.dataset import DarPInstance
from dialRL.utils import instance2world, indice2image_coordonates, distance
from dialRL.rl_train.environments import DarEnv

class DarSeqEnv(DarEnv):
    """Custom Environment that follows gym interface"""

    def __init__(self, size, target_population, driver_population, dataset=None, time_end=1400, max_step=10):

        self.dataset=dataset
        if self.dataset :
            super(DarSeqEnv, self).__init__(size, target_population, driver_population, time_end=1400, max_step=10)
            self.extremas, self.target_population, self.driver_population, self.time_end, self.depot_position, self.size = tabu_parse_info(self.dataset)

        else :
            super(DarSeqEnv, self).__init__(size, target_population, driver_population, time_end=1400, max_step=10)
            self.extremas = [-self.size, -self.size, self.size, self.size]
            self.depot_position = np.random.randint(0, self.size, (2))

        if False:
            print(self.extremas, self.target_population, self.driver_population, self.time_end, self.depot_position, self.size)

        self.max_step = max_step
        #self.driver_population*2 + self.target_population

        choix_id_target = True
        if not choix_id_target :
            self.action_space = spaces.Box(low=[self.extremas[0], self.extremas[1]],
                                           high=[self.extremas[2], self.extremas[3]],
                                           shape=(2,),
                                           dtype=np.float32)
        else :
            self.action_space = spaces.Discrete(self.target_population + 1)

        self.max_bloc_size = max(3 + self.target_population, 11)
        max_obs_value = max(self.target_population, self.extremas[2], self.extremas[3])
        self.observation_space = spaces.Box(low=-max_obs_value,
                                            high=max_obs_value,
                                            shape=(4 + 11*self.target_population + (3 + 6)*self.driver_population , ), #self.max_bloc_size*(1+self.target_population+self.driver_population)
                                            dtype=np.float)

        self.max_reward = int(1.5 * self.size)
        self.reward_range = (- self.max_reward, self.max_reward)
        self.time_step = 0
        self.current_episode = 0


    def get_info_vector(self):
        game_info = [self.time_step,
                     self.current_player,
                     self.depot_position[0],
                     self.depot_position[1]]
        return game_info


    def representation(self):
        # Agregate  world infrmations
        world_info = self.get_info_vector()

        # Agregate targets infrmations
        targets_info = []
        for target in self.targets:
            targets_info.append(target.get_info_vector())
        targets_info = np.concatenate(targets_info)

        drivers_info = []
        for driver in self.drivers:
            drivers_info.append(driver.get_info_vector())
        drivers_info = np.concatenate(drivers_info)

        world = np.concatenate([world_info, targets_info, drivers_info])

        return world

    def get_image_representation(self):
        # Return an image gathered from svg data
        size=self.size #max(abs(self.extremas[2] - self.extremas[0]), abs(self.extremas[3] - self.extremas[1]))

        d = draw.Drawing(size*2, size*2, origin='center', displayInline=False)
        for target in self.targets:
            # draw two nodes for pickup and delivery + An arrow connecting them
            if target.state > -1 :
                d.append(draw.Circle(target.pickup[0],target.pickup[1], 0.2,
                    fill='#8b5962', stroke_width=0.05, stroke='#555455'))
            else :
                d.append(draw.Circle(target.pickup[0],target.pickup[1], 0.2,
                        fill='red', stroke_width=0.05, stroke='black'))

            if target.state > 0 :
                d.append(draw.Circle(target.dropoff[0],target.dropoff[1], 0.2,
                    fill='#59598b', stroke_width=0.05, stroke='#555455'))
            else :
                d.append(draw.Circle(target.dropoff[0],target.dropoff[1], 0.2,
                        fill='blue', stroke_width=0.05, stroke='black'))

            d.append(draw.Line(target.pickup[0],target.pickup[1],
                               target.dropoff[0],target.dropoff[1],
                               stroke='green', stroke_width=0.02, fill='none'))

        for driver in self.drivers :
            d.append(draw.Circle(driver.position[0], driver.position[1], 0.3,
                    fill='yellow', stroke_width=0.2, stroke='black'))

        #d.setPixelScale(2)  # Set number of pixels per geometry unit
        d.setRenderSize(size*50, size*50)
        fo = tempfile.NamedTemporaryFile()
        d.savePng(fo.name)
        array_image = np.array(plt.imread(fo.name))
        fo.close()
        plt.imshow(array_image)
        return array_image


    def reset(self):
        self.instance = DarPInstance(size=self.size,
                                    population=self.target_population,
                                    drivers=self.driver_population,
                                    time_end=self.time_end,
                                    verbose=False)
        if self.dataset :
            self.instance.dataset_generation(self.dataset)
        else :
            self.instance.random_generation()

        # print('* Reset - Instance image : ', self.instance.image)
        self.targets = self.instance.targets.copy()
        self.drivers = self.instance.drivers.copy()

        self.current_player = 1
        # distance is -1 if wrong aiming. 0 if there is no start of game yet and x if aimed corectly
        self.distance = 0
        self.current_step = 0
        self.cumulative_reward = 0
        self.world = self.representation()
        self.last_aim = None
        self.last_cell = None
        self.short_log = ''
        return self._next_observation()


    def del_target(self, position):
        filter_fun = lambda x : x[0] == position[0] and x[1] == position[1]
        indices = [i for i in range(len(self.targets)) if filter_fun(self.targets[i])]
        for indice in indices:
            del self.targets[indice]


    def targets_to_go(self):
        count = [0 ,0, 0]
        for target in self.targets:
            count[target.state + 1] += 1
        return count


    def _next_observation(self):
        self.world = self.representation()
        obs = self.world
        return obs


    def _take_action(self, action):
        """ Action: destination point as an indice of the map vactor. (Ex: 1548 over 2500)
        """
        print(action)
        aiming_driver = self.drivers[self.current_player - 1]
        current_pos = aiming_driver.position

        #In case we aim an empty box
        if action == 0 :
            self.distance = 0
            self.short_log = 'Just do nothing'

        elif action > 0 and action <= self.target_population :
            aimed_target = self.targets[action - 1]
            self.last_aim = aimed_target.identity

            if aimed_target.state == 1 :
                self.distance = -3
                self.short_log = 'Aimed target already delivered'

            elif aimed_target.state == -1:
                result = aiming_driver.load(aimed_target)
                # Managed to load the target
                if result :
                    aimed_target.state = 0
                    aiming_driver.position = aimed_target.pickup

                    self.distance = distance(aimed_target.pickup, current_pos)
                    self.short_log = 'Aimed right, and pick up !'
                else :
                    self.distance = -4
                    self.short_log = 'Aimed  right but driver full'

            elif aimed_target.state == 0:
                result = aiming_driver.unload(aimed_target)
                if result :
                    aimed_target.state = 1
                    aiming_driver.position = aimed_target.dropoff
                    self.distance = distance(aimed_target.dropoff, current_pos)

                    self.short_log = 'Aimed right, and droping off !'

                else :
                    self.distance = -5
                    self.short_log = 'Aimed right BUT driver doesnt contain that target'
            else :
                self.distance = -6
                self.short_log = 'WRONG taret state'

        else :
            self.distance = -2
            self.short_log = 'Other wrong doing ? TODO'


    def reward(self, distance):
        return self.max_reward - distance #Linear distance
        #int(1.5 * self.size * (1 / distance))


    def step(self, action):
        # Curently the action is a heatmap ..
        # NOO now it's an indice in the list of action n*n

        self._take_action(action)
        self.current_step += 1

        if self.distance < 0:
            reward = -10 #-int(self.max_reward//2)
            done = False
        elif self.distance > 0:
            reward = self.reward(self.distance)
            done = False
            # update drivers turn
            self.current_player = ((self.current_player + 1 - 1) % (self.driver_population) ) + 1
        # End of simulation
        elif self.distance == 0 :
            reward = -5
            done = False
            self.current_player = ((self.current_player + 1 - 1) % (self.driver_population) ) + 1

        self.cumulative_reward += reward

        if self.targets_to_go()[2] == self.target_population :
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
        print(np.shape(self.world))

        if self.distance < 0 :
            print(f'Player {self.current_player} go lost ....')
        else :
            print(f'Player {self.current_player} aimed right')

        print('Crurrent player: ', self.current_player)
        print('Last aimed to : ', self.last_aim)
        print('Targets to go: ', self.targets_to_go())
        print('Cumulative reward : ', self.cumulative_reward)
        print('Additional  information : ', self.short_log)
        print('---------------------\n')


if __name__ == '__main__':
    data = './data/instances/cordeau2003/tabu1.txt'
    env = DarSeqEnv(size=4, target_population=5, driver_population=1, time_end=1400, max_step=5000, dataset=data)
    # env = DarSeqEnv(size=4, target_population=5, driver_population=2, time_end=1400, max_step=100, dataset=None)
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
