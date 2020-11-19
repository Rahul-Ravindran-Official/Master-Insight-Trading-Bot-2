import enum


class BBHSSSignal(enum.Enum):
    strong_buy = 5
    buy = 4
    hold = 3
    sell = 2
    strong_sell = 1
    no_action = 0
