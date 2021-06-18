from typing import Dict

import wandb
from hyperopt import fmin, tpe
from generation4.bot_backtester import BotBackTester, BotMetrics
from generation4.bots.bot_template import TradingBot

class BotOptimizer:

    max_evals: int
    _best_configuration: Dict
    _bot_metrics: BotMetrics
    _trading_bot_type: type(TradingBot)
    _optimization_config: Dict
    _wandb_config: Dict

    def __init__(self, trading_bot_type: type(TradingBot)):
        self.max_evals = 10
        self._best_configuration = {}
        self._bot_metrics = BotMetrics()
        self._trading_bot_type = trading_bot_type
        self._optimization_config = {}

        # Init WandB
        wandb.init(project='master-insight', entity='rahulravindran')
        self._wandb_config = wandb.config

    def _optimization_space(self) -> Dict:
        pass

    def _cost_function(self, bot_metrics: BotMetrics) -> float:
        pass

    def _args_template(self, args: Dict) -> Dict:
        pass

    # Universal Method
    def optimize(self):

        self._optimization_config = self._optimization_space()

        self._best_configuration = fmin(
            self._objective,
            self._optimization_config,
            algo=tpe.suggest,
            max_evals=self.max_evals,
        )

        print(f'Best Configuration: {self._best_configuration}')
        print(f'Full Best Configuration: {self._args_template(self._best_configuration)}')
        print(f'Best Configuration Yield: {self._objective(self._args_template(self._best_configuration))}')

    # Universal Method

    def _objective(self, args=None):

        self._wandb_config.algorithm = args['algorithm']
        wandb.log(args)

        bbt = BotBackTester(
            minimum_avg_trades_per_month=5,
            minimum_win_rate=0.1,
            stop_loss_pips=args['stop_loss_pips'],
            take_profit_pips=args['take_profit_pips'],
            pip_definition=0.0001,
            maximum_days_to_hold_per_trade=args['maximum_days_to_hold_per_trade'],
            bot_confidence_treshold=args['bot_confidence_treshold'],
            spread=0,
            bot_type=self._trading_bot_type,
            bot_metadata=args['bot_metadata']
        )

        self._bot_metrics = bbt.execute()

        wandb.log(self._bot_metrics.metrics())

        cost_function = self._cost_function(self._bot_metrics)

        wandb.log({"loss": cost_function})

        return cost_function

    def _log_dictionary(self, dict: Dict):
        wandb.log(dict)
