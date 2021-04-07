import numpy as np

import gym
from gym import spaces
import drawSvg as draw
import tempfile
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.image import imsave

from dialRL.dataset import tabu_parse_info, tabu_parse_best
from dialRL.dataset import DarPInstance
from dialRL.utils import instance2world, indice2image_coordonates, distance, instance2Image_rep, GAP_function, float_equality, coord2int
from dialRL.rl_train.environments import DarEnv

class DarSeqEnv(DarEnv):
    """Custom Environment that follows gym interface"""

    def __init__(self, size, target_population, driver_population, dataset=None, test_env=False, time_end=1400, max_step=10, verbose=0):

        self.dataset = dataset
        self.test_env = test_env
        self.max_step = max_step
        self.verbose = verbose
        if self.dataset :
            self.best_cost = tabu_parse_best(self.dataset)
            if self.test_env :
                super(DarSeqEnv, self).__init__(size, target_population, driver_population, time_end=time_end, max_step=self.max_step)
                self.extremas, self.target_population, self.driver_population, self.time_end, self.depot_position, self.size = tabu_parse_info(self.dataset)
            else :
                # Get info from dataset, to construct artificial data with those parameters
                super(DarSeqEnv, self).__init__(size, target_population, driver_population, time_end=time_end, max_step=self.max_step)
                self.extremas, self.target_population, self.driver_population, self.time_end, self.depot_position, self.size = tabu_parse_info(self.dataset)

        else :
            self.best_cost = 1000
            super(DarSeqEnv, self).__init__(size, target_population, driver_population, time_end=1400, max_step=max_step)
            self.extremas = [-self.size, -self.size, self.size, self.size]
            x = np.random.uniform(-self.size, self.size)
            y = np.random.uniform(-self.size, self.size)
            self.depot_position = np.array((x, y)) #, dtype=np.float16)

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
        self.obs_shape = 4 + 11*self.target_population + (3 + 6)*self.driver_population
        self.observation_space = spaces.Box(low=-max_obs_value,
                                            high=max_obs_value,
                                            shape=(4 + 11*self.target_population + (3 + 6)*self.driver_population , ), #self.max_bloc_size*(1+self.target_population+self.driver_population)
                                            dtype=np.float)

        self.max_reward = int(1.5 * self.size)
        self.reward_range = (- self.max_reward, self.max_reward)
        self.time_step = 0
        self.last_time_gap = 0
        self.current_episode = 0

        if self.verbose:
            print(' -- DarP Sequential Environment : -- ')
            for item in vars(self):
                print(item, ':', vars(self)[item])


    def get_info_vector(self):
        game_info = [coord2int(self.time_step),
                     self.current_player,
                     coord2int(self.depot_position[0]),
                     coord2int(self.depot_position[1])]
        return game_info

    def get_GAP(self):
        g = GAP_function(self.total_distance, self.best_cost)
        if g is None :
            return 300.
        else :
            return g

    def is_fit_solution(self):
        return int(self.targets_to_go()[4] == self.target_population)

    def representation(self):
        # Agregate  world infrmations
        world_info = self.get_info_vector()
        # print('world: ', world_info)

        # Agregate targets infrmations
        targets_info = []
        for target in self.targets:
            targets_info.append(target.get_info_vector())
        targets_info = np.concatenate(targets_info)
        # print('targets_info: ', targets_info)

        drivers_info = []
        for driver in self.drivers:
            drivers_info.append(driver.get_info_vector())
        drivers_info = np.concatenate(drivers_info)
        # print('drivers_info: ', drivers_info)

        world = np.concatenate([world_info, targets_info, drivers_info])

        return world

    def get_image_representation(self):
        image = instance2Image_rep(self.targets, self.drivers, self.size, time_step=self.time_step)
        return image


    def reset(self):
        self.instance = DarPInstance(size=self.size,
                                    population=self.target_population,
                                    drivers=self.driver_population,
                                    depot_position=self.depot_position,
                                    extremas=self.extremas,
                                    time_end=self.time_end,
                                    verbose=False)
        if self.test_env and self.dataset:
            self.instance.dataset_generation(self.dataset)
        else :
            self.instance.random_generation()

        # print('* Reset - Instance image : ', self.instance.image)
        self.targets = self.instance.targets.copy()
        self.drivers = self.instance.drivers.copy()

        # It is important to let time step at target forks as well,
            #in order to let possibility for driver to wake up after waiting
        self.target_times = []
        for target in self.targets :
            self.target_times.append(target.start_fork[0])
            self.target_times.append(target.start_fork[1])
            self.target_times.append(target.end_fork[0])
            self.target_times.append(target.end_fork[1])

        self.next_players = [i for i in range(2, self.driver_population+1)]
        self.current_player = 1
        # distance is -1 if wrong aiming. 0 if there is no start of game yet and x if aimed corectly
        self.time_step = 0
        self.distance = 0
        self.total_distance = 0
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
        count = [0, 0, 0, 0, 0]
        for target in self.targets:
            count[target.state + 2] += 1
        return count


    def _next_observation(self):
        self.world = self.representation()
        obs = self.world
        return obs


    def _take_action(self, action):
        """ Action: destination point as an indice of the map vactor. (Ex: 1548 over 2500)
        """
        aiming_driver = self.drivers[self.current_player - 1]
        current_pos = aiming_driver.position

        #In case we aim an empty box
        if action == 0 :
            self.distance = 0
            self.short_log = 'Just do nothing'
            aiming_driver.set_target(None, self.time_step)

        elif action > 0 and action <= self.target_population :
            aimed_target = self.targets[action - 1]
            self.last_aim = aimed_target.identity

            if aimed_target.state == 2 :
                self.distance = -3
                self.short_log = 'Aimed target already delivered'

            elif aimed_target.state == -2:
                result = aiming_driver.set_target(aimed_target, self.time_step)
                # Managed to load the target
                if result :
                    self.distance = distance(aiming_driver.position, aiming_driver.destination)
                    self.short_log = 'Aimed right, going for pick up !'
                else :
                    self.distance = -4
                    self.short_log = 'Aimed free target but couldn"t load target (driver full, or wrong time window)'

            elif aimed_target.state == 0:
                result = aiming_driver.set_target(aimed_target, self.time_step)
                if result :
                    self.distance = distance(aiming_driver.position, aiming_driver.destination)
                    self.short_log = 'Aimed right, and goiong for dropoff !'

                else :
                    self.distance = -5
                    self.short_log = 'Aimed right BUT driver doesnt contain that target'
            else :
                self.distance = -6
                self.short_log = 'That target is already been taken care of'

        else :
            self.distance = -2
            self.short_log = 'Other wrong doing ? TODO'


    def reward(self, distance):
        return self.max_reward - distance #Linear distance
        #int(1.5 * self.size * (1 / distance))


    def update_time_step(self):
        # Should other types of events be added here ?
            # Such as the end of the game event
        events_in = []
        for driver in self.drivers:
            if driver.destination is not None :
                events_in.append(self.time_step + distance(driver.position, driver.destination))
        events_in = events_in + self.target_times
        events_in = [t for t in events_in if t>self.time_step]
        self.last_time_gap = min(events_in) - self.time_step
        self.time_step = min(events_in)


    def update_drivers_positions(self):
        if self.last_time_gap > 0:
            for driver in self.drivers :
                if driver.destination is not None :
                    d = distance(driver.position, driver.destination)
                    if float_equality(self.last_time_gap, d, eps=0.001):
                        # Driver arraving to destination
                        driver.move(driver.destination)
                        if driver.order == 'picking':
                            result = driver.load(driver.target, self.time_step)
                            if not result :
                                raise "Error while loading the target, it is intended to be pickupable"

                        elif driver.order == 'dropping':
                            result = driver.unload(driver.target, self.time_step)
                            if not result :
                                raise "Error while unloading the target, it is intended to be droppable"

                        # reset the driver on waiting list
                        driver.set_target(None, self.time_step)
                        # self.next_players.append(driver.identity)

                    elif self.last_time_gap < d:
                        # lx + (1-l)x with l=d'/d
                        d = distance(driver.position, driver.destination)
                        lam = self.last_time_gap / d
                        new_pos = (1. - lam) * np.array(driver.position) + (lam * np.array(driver.destination))
                        # print('new_pos1: ', new_pos)
                        # lam = np.array(self.last_time_gap, dtype=np.float32) / np.array(d, dtype=np.float32)
                        # new_pos = (1. - lam) * np.array(driver.position, dtype=np.float32) + (lam) * np.array(driver.destination, dtype=np.float32)
                        # print('new_pos32: ', new_pos)
                        # lam = np.array(self.last_time_gap, dtype=np.float16) / np.array(d, dtype=np.float16)
                        # new_pos = (1. - lam) * np.array(driver.position, dtype=np.float16) + (lam) * np.array(driver.destination, dtype=np.float16)
                        # print('new_pos16: ', new_pos)
                        # # print('Comparaison of types: ', np.array(driver.position, dtype=np.float32), driver.position)
                        # print('lam: ', lam)
                        # print('distance to target', d)
                        # print('distance to new pose', distance(new_pos, driver.position))
                        # print('distance to new pose with array', distance(np.array(new_pos), np.array(driver.position)))
                        # print('Multiplied distance :', distance(m * new_pos, m * np.array(driver.position))/m)
                        # print('Time gap:', self.last_time_gap)
                        # print('equ 0.01', float_equality(distance(new_pos, np.array(driver.position)), self.last_time_gap, eps=0.01))
                        # print('equ 0.001', float_equality(distance(new_pos, np.array(driver.position)), self.last_time_gap, eps=0.001))
                        # print('equ 0.0001', float_equality(distance(new_pos, np.array(driver.position)), self.last_time_gap, eps=0.0001))
                        # print('equ 0.00001', float_equality(distance(new_pos, np.array(driver.position)), self.last_time_gap, eps=0.00001))
                        # exit()
                        if not float_equality(distance(new_pos, driver.position), self.last_time_gap, eps=0.001):
                            raise 'Distance float problem ? Here the distance to new position is different to time passing !'
                        driver.move(new_pos)

                    else :
                        raise "Error in updating drivers position. distance to destination:" + \
                        str(d) + "last time gap:" + str(self.last_time_gap)


    def step(self, action):
        # Action is the selected target id to handle (Eiither pick of drop)
        self._take_action(action)
        self.current_step += 1

        # Current time step need to be updated -> Driver move as well
        if not self.next_players and self.distance >= 0:
            while len(self.next_players) == 0 :
                # If no drivers turn, activate time steps
                if False :
                    image = self.get_image_representation()
                    imsave('./data/rl_experiments/test/' + str(env.current_step) + 'a.png', image)
                self.update_time_step()
                self.update_drivers_positions()
                if False :
                    image = self.get_image_representation()
                    imsave('./data/rl_experiments/test/' + str(env.current_step) + 'b.png', image)
                for driver in self.drivers :
                    if driver.destination is None :
                        # Charge all players that may need a new destination
                        self.next_players.append(driver.identity)

        # Generate reward from distance
        if self.distance < 0:
            reward = -1 #-int(self.max_reward//2)
            done = False
        elif self.distance > 0:
            reward = self.reward(self.distance)
            done = False
            self.total_distance += self.distance
        elif self.distance == 0 :
            reward = -1
            done = False

        # Update current player (if last action was successfull)
        if self.distance >=0 :
            self.current_player = self.next_players.pop()

        self.cumulative_reward += reward

        if self.targets_to_go()[4] == self.target_population :
            # High reward for finishing the game
            reward += 100
            done = True
        if self.current_step >= self.max_step or self.time_step >= self.time_end :
            reward -= 100
            done = True

        if done:
            self.current_episode += 1
            # print('End of episode, total reward :', self.cumulative_reward)


        obs = self._next_observation()

        info = {
            'delivered': self.targets_to_go()[4],
            'GAP': self.get_GAP(),
            'fit_solution': self.is_fit_solution()
        }
        # Last element is info (dict)
        return obs, reward, done, info


    def render(self, mode='classic'):
        print('\n--------------------- [Step', self.current_step, ']')

        print('World: ')
        # print(self.world)
        print(np.shape(self.world))

        if self.distance < 0 :
            print(f'Player {self.current_player} go lost ....')
        else :
            print(f'Player {self.current_player} aimed right')

        print('Crurrent time step: ', self.time_step)
        print('Crurrent player: ', self.current_player)
        print('Next player: ', self.next_players)
        print('Last aimed to : ', self.last_aim)
        print('Targets to go: ', self.targets_to_go())
        print('Cumulative reward : ', self.cumulative_reward)
        print(' Cumulative distance :', self.total_distance)
        print('Additional  information : ', self.short_log)
        print('GAP to best known solution: ', GAP_function(self.total_distance, self.best_cost))
        print('Is this a fit solution ? -> ', self.is_fit_solution())
        print('---------------------\n')


if __name__ == '__main__':
    data = './data/instances/cordeau2003/tabu1.txt'
    # data = None
    env = DarSeqEnv(size=4,
                    target_population=2,
                    driver_population=2,
                    time_end=1400,
                    max_step=5000,
                    dataset=data,
                    test_env=False)
    # env = DarSeqEnv(size=4, target_population=5, driver_population=2, time_end=1400, max_step=100, dataset=None)
    cumulative_reward = 0
    observation = env.reset()
    env.render()
    for t in range(5000):
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        cumulative_reward += reward
        print('Cumulative reward : ', cumulative_reward, ' (', reward, ')')
        env.render()

        if done:
            print("\n ** \t Episode finished after {} time steps".format(t+1))
            break
    env.close()
