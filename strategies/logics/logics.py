
class Logic(object):

    def __init__(self, logic_str, name):
        self.logic_str = logic_str
        self.name = name

    def create_signal(self, candles):
        exec(self.logic_str, None, {"p1h": candles})
        return candles


class StrategiesLogics(object):

    @classmethod
    def create_logic(cls, file_name):
        with open(file_name, "r") as f:
            logic_str = f.read()
            return Logic(logic_str, "test")


if __name__ == "__main__":

    from datetime import datetime
    import pytz
    from strategies.candles.ftx_candle_generator import FtxCandleGenerator
    from strategies.generator import PnlCalculator
    start_timestamp = datetime(2021, 1, 1, tzinfo=pytz.UTC).timestamp()
    end_timestamp = datetime(2021, 7, 1, tzinfo=pytz.UTC).timestamp()
    generator = FtxCandleGenerator("BTC-PERP", start_timestamp, end_timestamp)
    df = generator.generate()
    print(df)

    logic_no = 0
    logic = StrategiesLogics.create_logic(f"{logic_no:03}.txt")

    df = logic.create_signal(df)
    print(df)

    df = PnlCalculator.calculate(generator, logic)
    print(df)
