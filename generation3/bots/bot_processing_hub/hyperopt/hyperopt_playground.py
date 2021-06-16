import unittest
from hyperopt import hp
from hyperopt import fmin, tpe

from generation3.bots.knn_bot import hyperopt_knn_objective, knn_func
from generation3.bots.nn_bot import NNBot


class HyperoptLab(unittest.TestCase):

    @staticmethod
    def objective(args):
        case, val = args
        if case == 'case 1':
            return val
        else:
            return val ** 2

    def test_simple(self):

        # define a search space
        space = hp.choice('a', [
            ('case 1', 1 + hp.lognormal('c1', 0, 1)),
            ('case 2', hp.uniform('c2', -10, 10))
        ])

        best = fmin(HyperoptLab.objective, space, algo=tpe.suggest,
                    max_evals=100)

        self.assertEqual(True, True)

    # ------------------------------
    # KNN BOT
    # ------------------------------

    def test_knn_bot(self):

        # define a search space
        space = hp.choice('a', [
            {
                'type': {
                    'buy_signal': {
                        'neighbours': hp.randint('neighbours_buy_dist', 15) + 1,
                        'treshold': {
                            'low': hp.uniform('buy_treshold_low_dist', 0, 1),
                            'high': 1
                        },
                        'weights': hp.choice('weights_dist', ['uniform', 'distance'])
                    },
                    'sell_signal': {
                        'neighbours': hp.randint('neighbours_sell_dist', 15) + 1,
                        'treshold': {
                            'low': hp.uniform('sell_treshold_low_dist', 0, 1),
                            'high': 1
                        },
                    }
                },
                'pca_components': hp.randint('pca_components_dist', 15),
            }
        ])

        best = fmin(hyperopt_knn_objective, space, algo=tpe.suggest,
                    max_evals=1000)

        print(best)

    def test_knn_tuned_bot(self):
        config = {
            'neighbours': 9 + 1,
            'buy_treshold': {
                'low': 0.49570496767640104,
                'high': 1
            },
            'sell_treshold': {
                'low': 0.29047395758118,
                'high': 1
            }
        }

        print(knn_func(config))

    # ------------------------------
    # NN BOT
    # ------------------------------

    def test_nn_bot(self):

        # define a search space
        space = hp.choice('a', [
            {
                'type': {
                    'buy_signal': {
                        'epochs': hp.randint('epochs_dist', 15) + 1,
                        'treshold': {
                            'low': hp.uniform('buy_treshold_low_dist', 0, 1),
                            'high': 1
                        },
                        'weights': hp.choice('weights_dist', ['uniform', 'distance']),
                        'nn':{
                            'l1-nodes': hp.choice(
                                'buy_l1_nodes_dist',
                                [
                                    1, 2, 3, 4, 5, 10, 15, 20, 30, 50, 75, 100
                                ]
                            ),
                            'l2-nodes': hp.choice(
                                'buy_l2_nodes_dist',
                                [
                                    1, 2, 3, 4, 5, 10, 15, 20, 30, 50, 75, 100
                                ]
                            ),
                            'l3-nodes': hp.choice(
                                'buy_l3_nodes_dist',
                                [
                                    1, 2, 3, 4, 5, 10, 15, 20, 30, 50, 75, 100
                                ]
                            ),
                            'l4-nodes': hp.choice(
                                'buy_l4_nodes_dist',
                                [
                                    1, 2, 3, 4, 5, 10, 15, 20, 30, 50, 75, 100
                                ]
                            ),
                            'class_weight': {
                                'class-weight-0': hp.uniform('buy_class_weight_0_dist', 0, 1),
                                'class-weight-1': hp.uniform('buy_class_weight_1_dist', 0, 1)
                            }
                        }
                    },
                    'sell_signal': {
                        'epochs': hp.randint('epochs_sell_dist', 15) + 1,
                        'treshold': {
                            'low': hp.uniform('sell_treshold_low_dist', 0, 1),
                            'high': 1
                        },
                        'nn': {
                            'l1-nodes': hp.choice(
                                'sell_l1_nodes_dist',
                                [
                                    1, 2, 3, 4, 5, 10, 15, 20, 30, 50, 75, 100
                                ]
                            ),
                            'l2-nodes': hp.choice(
                                'sell_l2_nodes_dist',
                                [
                                    1, 2, 3, 4, 5, 10, 15, 20, 30, 50, 75, 100
                                ]
                            ),
                            'l3-nodes': hp.choice(
                                'sell_l3_nodes_dist',
                                [
                                    1, 2, 3, 4, 5, 10, 15, 20, 30, 50, 75, 100
                                ]
                            ),
                            'l4-nodes': hp.choice(
                                'sell_l4_nodes_dist',
                                [
                                    1, 2, 3, 4, 5, 10, 15, 20, 30, 50, 75, 100
                                ]
                            ),
                            'class_weight': {
                                'class-weight-0': hp.uniform('sell_class_weight_0_dist', 0, 1),
                                'class-weight-1': hp.uniform('sell_class_weight_1_dist', 0, 1)
                            }
                        }
                    }
                },
                'pca': {
                    'pca_components_buy': hp.randint(
                        'pca_components_buy_dist', 10
                    ),
                    'pca_components_sell': hp.randint(
                        'pca_components_sell_dist', 10
                    ),
                }
            }
        ])

        best = fmin(
            NNBot.hyperopt_nn_objective,
            space,
            algo=tpe.suggest,
            max_evals=1000
        )

        print(best)


if __name__ == '__main__':
    unittest.main()
