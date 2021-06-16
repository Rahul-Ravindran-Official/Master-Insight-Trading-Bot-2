# Bot Performances

## Goal
1. Win Count / Total Trades >= 60%
2. Average Profit / Average Loss >= 2
3. Trade Count / Ideal_Trade_Count=52 >= 75% 

## Theoretical Bot

### spread=3.5

```python
    b_t_x, b_t_y, b_v_x, true_buy_signal, _, _ = RefinedData.get_dataset_common(
        None,
        buy_sell_dataset='b'
    )

    s_t_x, s_t_y, s_v_x, true_sell_signal, _, _ = RefinedData.get_dataset_common(
        None,
        buy_sell_dataset='s'
    )

    bbt = BotBackTester(
        TheoreticalBot(
            None, None, None,
            None, None, None,
            {
                'true_buy_signal': true_buy_signal,
                'true_sell_signal': true_sell_signal
            }
        ),
        list(b_v_x["Adj Close"]),
        buy_treshold=(0.5, 1.0),
        sell_treshold=(0.9, 1.0),
        spread=3.5,
        lot_size=0.1,
        pip_definition=0.0001,
        profit_per_pip_per_lot=10
    )

    bbt.back_test()
    bbt.print_stats()
```

Price Gained : 0.600490311908722
Pips Gained : 6004.90311908722
Profits Earned : 6004.90311908722
Days Traded : 336
Total Trades : 52
Profit Counts : 49
Loss Count : 3
Max Price Profit : 0.03884786214828491
Max Price Loss : -0.00027215633392333984

---

### spread=0

```python
    b_t_x, b_t_y, b_v_x, true_buy_signal, _, _ = RefinedData.get_dataset_common(
        None,
        buy_sell_dataset='b'
    )

    s_t_x, s_t_y, s_v_x, true_sell_signal, _, _ = RefinedData.get_dataset_common(
        None,
        buy_sell_dataset='s'
    )

    bbt = BotBackTester(
        TheoreticalBot(
            None, None, None,
            None, None, None,
            {
                'true_buy_signal': true_buy_signal,
                'true_sell_signal': true_sell_signal
            }
        ),
        list(b_v_x["Adj Close"]),
        buy_treshold=(0.5, 1.0),
        sell_treshold=(0.9, 1.0),
        spread=0,
        lot_size=0.1,
        pip_definition=0.0001,
        profit_per_pip_per_lot=10
    )

    bbt.back_test()
    bbt.print_stats()
```

Price Gained : 0.6186903119087218
Pips Gained : 6186.903119087217
Profits Earned : 6186.903119087218
Days Traded : 336
Total Trades : 52
Profit Counts : 52
Loss Count : 0
Max Price Profit : 0.03919786214828491
Max Price Loss : inf

## KNN Bot
