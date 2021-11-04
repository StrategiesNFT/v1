import numpy as np


class PnlCalculator(object):

    @classmethod
    def calculate(cls, candle_generator, logic):
        df = candle_generator.generate()
        df = logic.create_signal(df)
        # TODO Refactoring.
        df["position"] = df["signal"].ffill().shift(1).fillna(0)
        df["entry_price"] = np.nan
        df.loc[df["position"].diff(1) != 0, "entry_price"] = df["o"]
        df["entry_price"] = df["entry_price"].ffill()
        df["exit_price"] = np.nan
        df.loc[df["position"].diff(1).shift(-1) != 0, "exit_price"] = df["c"]
        df["pnl"] = 0
        df.loc[~np.isnan(df["exit_price"]), "pnl"] = df["position"] * (df["exit_price"] - df["entry_price"]) / df["entry_price"]
        df["tmp_pnl"] = 0
        df.loc[df["position"] == 1, "tmp_pnl"] = (df["c"] - df["entry_price"]) / df["entry_price"]
        df.loc[df["position"] == -1, "tmp_pnl"] = -(df["c"] - df["entry_price"]) / df["entry_price"]
        df.loc[~np.isnan(df["exit_price"]), "tmp_pnl"] = 0
        df["balance"] = df["pnl"].cumsum() + df["tmp_pnl"]
        df["balance"] = df["balance"].ffill()
        return df
