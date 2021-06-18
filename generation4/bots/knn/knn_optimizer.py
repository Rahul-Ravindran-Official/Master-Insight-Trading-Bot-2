from typing import Dict

from generation4.bot_backtester import BotMetrics
from generation4.bot_optimizer import BotOptimizer
from generation4.bots.knn.knn_bot import KnnTradingBot

from hyperopt import hp

class KnnBotOptimizer(BotOptimizer):

    def __init__(self):
        super().__init__(KnnTradingBot)
        self.max_evals = 5000

    # def _optimization_space(self) -> Dict:
    #     return {
    #         'algorithm': 'KNN',
    #         'stop_loss_pips': hp.randint('stop_loss_pips', 200) + 1,
    #         'take_profit_pips': hp.randint('take_profit_pips', 200) + 1,
    #         'maximum_days_to_hold_per_trade': hp.randint('maximum_days_to_hold_per_trade', 10) + 1,
    #         'bot_confidence_treshold': hp.uniform('bot_confidence_treshold', 0, 1),
    #         'bot_metadata': {
    #             'n_neighbors': hp.randint('n_neighbors', 20) + 1,
    #             'weights': hp.choice('weights', ['uniform', 'distance'])
    #         }
    #     }

    def _optimization_space(self) -> Dict:
        return {
            'algorithm': 'KNN',
            'stop_loss_pips': 10,
            'take_profit_pips': 100,
            'maximum_days_to_hold_per_trade': hp.randint('maximum_days_to_hold_per_trade', 10) + 1,
            'bot_confidence_treshold': hp.uniform('bot_confidence_treshold', 0, 1),
            'bot_metadata': {
                'n_neighbors': hp.randint('n_neighbors', 20) + 1,
                'weights': hp.choice('weights', ['uniform', 'distance'])
            }
        }

    def _args_template(self, args: Dict):
        return {
            'algorithm': 'KNN',
            'stop_loss_pips': args['stop_loss_pips'],
            'take_profit_pips': args['take_profit_pips'],
            'maximum_days_to_hold_per_trade': args['maximum_days_to_hold_per_trade'],
            'bot_confidence_treshold': args['bot_confidence_treshold'],
            'bot_metadata': {
                'n_neighbors': args['n_neighbors'],
                'weights': args['weights']
            }
        }

    def _cost_function(self, bot_metrics: BotMetrics) -> float:
        return -bot_metrics.profit_pips_earned

if __name__ == "__main__":
    KnnBotOptimizer().optimize()
