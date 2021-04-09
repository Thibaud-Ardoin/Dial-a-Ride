import numpy as np

from dialRL.strategies import BaseStrategy
from dialRL.utils import distance


class NNStrategy(BaseStrategy):
    def __init__(self, **kwarg):
        super().__init__(**kwarg)

    def action_choice(self, observation):
        player_nb = self.env.current_player
        player = self.env.drivers[player_nb - 1]
        choice = (None, np.inf)
        for target in self.env.targets :

            # Potatial pickup
            if target.state == -2 :
                if player.can_load(target, self.env.time_step) :
                    d = distance(player.position, target.pickup)
                    if choice[1] > d:
                        choice = (target, d)

            # Potential Dropoff
            elif target.state == 0 :
                if player.can_unload(target, self.env.time_step) :
                    d = distance(player.position, target.pickup)
                    if choice[1] > d:
                        choice = (target, d)
        if choice[0] is None :
            return 0
        return choice[0].identity


if __name__ == '__main__':
    strat = NNStrategy(size=4,
                        target_population=2,
                        driver_population=2,
                        reward_function='ConstantReward',
                        time_end=1400,
                        max_step=5000,
                        dataset='./data/instances/cordeau2003/tabu6.txt',
                        test_env=True)

    strat.run()
