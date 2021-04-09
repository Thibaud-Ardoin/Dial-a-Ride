from dialRL.strategies import BaseStrategy

class RandomStrategy(BaseStrategy):
    def __init__(self, **kwarg):
        super().__init__(**kwarg)

    def action_choice(self, observation):
        return self.env.action_space.sample()


if __name__ == '__main__':
    strat = RandomStrategy(size=4,
                        target_population=2,
                        driver_population=2,
                        reward_function='HybridProportionalReward',
                        time_end=1400,
                        max_step=5000,
                        dataset='./data/instances/cordeau2003/tabu1.txt',
                        test_env=True)

    strat.run()
