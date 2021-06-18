from typing import Dict

from generation4.bot_backtester import BotMetrics
from generation4.bot_optimizer import BotOptimizer
from generation4.bots.knn.knn_bot import KnnTradingBot

from hyperopt import hp

from generation4.bots.svm.svm_bot import SVMTradingBot


class SVMBotOptimizer(BotOptimizer):

    def __init__(self):
        super().__init__(SVMTradingBot)
        self.max_evals = 10

    def _optimization_space(self) -> Dict:
        return {
            'algorithm': 'SVM',
            'stop_loss_pips': hp.randint('stop_loss_pips', 200) + 1,
            'take_profit_pips': hp.randint('take_profit_pips', 200) + 1,
            'maximum_days_to_hold_per_trade': hp.randint(
                'maximum_days_to_hold_per_trade', 10) + 1,
            'bot_confidence_treshold': hp.uniform('bot_confidence_treshold', 0,
                                                  1),
            'bot_metadata': {
                'kernel': hp.choice('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
            }
        }

    def _args_template(self, args: Dict):
        return {
            'algorithm': 'SVM',
            'stop_loss_pips': args['stop_loss_pips'],
            'take_profit_pips': args['take_profit_pips'],
            'maximum_days_to_hold_per_trade': args[
                'maximum_days_to_hold_per_trade'],
            'bot_confidence_treshold': args['bot_confidence_treshold'],
            'bot_metadata': {
                'kernel': args['kernel']
            }
        }

    def _cost_function(self, bot_metrics: BotMetrics) -> float:
        return -bot_metrics.profit_pips_earned


if __name__ == "__main__":
    SVMBotOptimizer().optimize()

    # score = SVMBotOptimizer()._objective(
    #     {'stop_loss_pips': 172, 'take_profit_pips': 19,
    #      'maximum_days_to_hold_per_trade': 1,
    #      'bot_confidence_treshold': 0.34900918680332377,
    #      'bot_metadata': {'kernel': 'sigmoid'}})
    #
    # print(f'Best Configuration Yield: {score}')


