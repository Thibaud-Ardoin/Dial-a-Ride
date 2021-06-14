import json
from icecream import ic

from dialRL.strategies import BaseStrategy

class CompleteRoute(BaseStrategy):
    def __init__(self, solution_file, **kwarg):
        self.parse_solution(solution_file)
        super().__init__(**kwarg)
        self.routes_status = [0] * self.env.driver_population


    def action_choice(self, observation=None):
        # # test shit
        # if self.env.time_step > 102 :
        #     exit()
        player_id = self.env.current_player
        player = self.env.drivers[player_id - 1]

        if player_id > len(self.routes):
            return 0
        if self.routes_status[player_id - 1] >= len(self.routes[player_id - 1]) :
            return 0

        next_node = self.routes[player_id - 1][self.routes_status[player_id - 1]]
        target_id = self.node2target(next_node)
        target = self.env.targets[target_id - 1]

        if player.can_load(target, self.env.time_step) :
            self.routes_status[player_id - 1] += 1
            return target_id
        elif player.can_unload(target, self.env.time_step) :
            self.routes_status[player_id - 1] += 1
            return target_id

        return 0


    def parse_solution(self, solution_file):
        with open(solution_file, 'r') as solfile:
            data = solfile.read()
            obj = json.loads(data)
        self.routes = obj['routes']

    def parcours(self):
        for t in self.routes:
            print(t)
            for node in t:
                target = self.node2target(node)
                print(target)
        exit()

    def node2target(self, node):
        if node > self.env.target_population:
            return node - self.env.target_population
        else :
            return node


if __name__ == '__main__':
    strat = CompleteRoute(
        solution_file='/home/tibo/Documents/Prog/EPFL/own/dialRL/strategies/logs/darp/rf/4443b2f25d373531/RF0-0_soln.json',
        #'/home/tibo/Documents/Prog/EPFL/own/dialRL/strategies/logs/darp/rf/4443b2f25d373531/RF11-pr01_soln.json',
        #/home/tibo/Documents/Prog/EPFL/own/dialRL/strategies/logs/darp/rf/4443b2f25d373531/RF7-service_test3_soln.json',
        size=4,
        target_population=2,
        driver_population=2,
        reward_function='HybridProportionalReward',
        time_end=1400,
        max_step=5000,
        dataset='/home/tibo/Documents/Prog/EPFL/own/dialRL/strategies/data/DARP_cordeau/0.txt',
        test_env=True,
        recording=True)

    strat.run()
