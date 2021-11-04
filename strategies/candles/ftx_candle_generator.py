import ccxt
import pandas as pd
import pytz
import time
from datetime import datetime
from strategies.candles import CandleGeneratorBase


class FtxCandleGenerator(CandleGeneratorBase):

    def __init__(self, symbol, start_timestamp, end_timestamp):
        self.timeframe = "1h"
        self.symbol = symbol
        self.start_timestamp = start_timestamp
        self.end_timestamp = end_timestamp
        self.candles = None

    def generate(self, regenerate=False):
        ftx = ccxt.ftx()
        df = []
        since = self.start_timestamp * 1000
        until = self.end_timestamp * 1000
        while True:
            tmp = ftx.fetch_ohlcv(symbol=self.symbol, timeframe=self.timeframe, since=since)
            df.append(pd.DataFrame(tmp))
            if tmp[-1][0] > until:
                break
            else:
                since = tmp[-1][0] + 1
                time.sleep(0.3)
        df = pd.concat(df)
        df.columns = ["ts", "o", "h", "l", "c", "v"]
        df = df[df["ts"] < until]
        df["ts"] = (df["ts"] / 1000).round()
        df = df.set_index("ts")
        self.candles = df
        return self.candles


if __name__ == "__main__":
    start_timestamp = datetime(2021, 1, 1, tzinfo=pytz.UTC).timestamp()
    end_timestamp = datetime(2021, 7, 1, tzinfo=pytz.UTC).timestamp()
    generator = FtxCandleGenerator("BTC-PERP", start_timestamp, end_timestamp)
    df = generator.generate()
    print(df)
