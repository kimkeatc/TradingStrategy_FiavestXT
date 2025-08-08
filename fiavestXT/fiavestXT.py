#!/usr/bin/env python

# -*- coding: utf-8 -*-

# Import standard libraries
from os.path import dirname, exists, join
from datetime import datetime
import threading
import logging
import json
import sys

# Import third-party libraries
from malaysia_stock_market_miscellaneous.misc import get_candle_ratios, number_of_bid_diff
from klsescreener.misc import chunks, dec_performance
from klsescreener.klsescreener import KLSEScreener
from klsescreener.stock import Stock
import pandas
pandas.set_option('display.max_columns', None)

NUMBER_OF_THREADS = 16

MIN_PRICE = 0.3
MAX_PRICE = 2.0
MARKET = "Main Market"
MIN_VOLUME = 40000
MINIMUM_BID_CHANGES = 4
MIN_BODY_RATIO = 60.0


class FiavestXT:

    def __init__(self):
        self.record_filename = "records.json"
        self.record_filepath = join(dirname(__file__), self.record_filename)

        self.record_tablename = "records.xlsx"
        self.record_tablepath = join(dirname(__file__), self.record_tablename)

        self.records = self.load_records()

    def load_records(self) -> dict:
        records = {}
        if not exists(self.record_filepath):
            with open(self.record_filepath, "w") as fh:
                json.dump(obj=records, fp=fh, indent=4, sort_keys=True)

        with open(self.record_filepath, "r") as fh:
            records = json.load(fp=fh)
        return records

    def save_records(self):
        with open(self.record_filepath, "w") as fh:
            json.dump(obj=self.records, fp=fh, indent=4, sort_keys=True)

    @dec_performance(log=logging.info)
    def get_pillars(self, days: int = 15, min_price: float = MIN_PRICE, max_price: float = MAX_PRICE, market: str = MARKET, min_volume: int = MIN_VOLUME, minimum_bid_changes: int = MINIMUM_BID_CHANGES, min_body_ratio: float = MIN_BODY_RATIO):

        dataframe = KLSEScreener().screener()

        def mThread(stockcodes: list[str] = [], results: dict = {}, number_of_threads: int = NUMBER_OF_THREADS) -> dict:
            threads = []
            stockcodes = chunks(iterable=stockcodes, n=number_of_threads)
            for stockcode in stockcodes:
                threads.append(threading.Thread(
                    target=sThread,
                    args=[stockcode, results]
                ))
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            logging.info("All threads have completed.")
            return results
    
        def sThread(stockcodes: list[str] = [], results: dict = {}) -> dict:
            
            length_stockcodes = len(stockcodes)
            logging.info(f"Total number of available stockcodes: {length_stockcodes}")

            for index, stockcode in enumerate(stockcodes, start=1):
                df = dataframe[dataframe["Code"] == stockcode].iloc[0]
                stockname, category, chart_link = df["Name"], df["Category"], df["KLSEScreener Chart"]
                logging.debug(f"[{index:02d}/{length_stockcodes:02d}] Processing stock code: {stockcode:5s} name: {stockname:8s} category: {category}")

                # Filters - general
                if market not in category:
                    continue

                df_historical_data_1D = Stock(stockcode=stockcode).historical_data_1D()

                df = df_historical_data_1D.copy()
  
                last_transacted_date = datetime.combine(date=df.iloc[0]["Date"], time=datetime.min.time())
                days_differences = (datetime.now() - last_transacted_date).days
                if days_differences > 5:
                    continue

                df = df.head(n=days)
                df = df[df["v"] > min_volume]
                df = df[(df["c"] >= min_price) & (df["c"] <= max_price)]

                # Filters - based on the candle
                indexex_to_be_remove = []
                for index, row in df.iterrows():
                    if row["c"] <= row["o"]:
                        indexex_to_be_remove.append(index)
                    elif number_of_bid_diff(a=row["o"], b=row["c"]) < minimum_bid_changes:
                        indexex_to_be_remove.append(index)
                    else:
                        _, _, body_ratio, _ = get_candle_ratios(oPrice=row["o"], hPrice=row["h"], lPrice=row["l"], cPrice=row["c"])
                        if body_ratio < min_body_ratio:
                            indexex_to_be_remove.append(index)
                df = df.drop(index=indexex_to_be_remove)
      
                if df.empty:
                    continue

                # Remaining
                df = df_historical_data_1D.iloc[: df.index[0] + 1]
                df = df[::-1]
                df = df.reset_index(drop=True)

                # Filters - daily transaction
                break_support, pillar, half_of_pillar_price = False, None, None
                for index, row in df.iterrows():
                    if index == 0:
                        pillar = row
                        half_of_pillar_price = (pillar["c"] + pillar["o"]) / 2
                    elif index <= 1:
                        if half_of_pillar_price > row["c"]:
                            break_support = True
                            break
                    elif index <= 5:
                        if pillar["o"] > row["c"]:
                            break_support = True
                            break
                if break_support is True:
                    continue

                result = results.setdefault(stockcode, {})
                date = str(df["Date"].to_list()[0])
                logging.info(f"Added stock code {stockcode} {stockname:8s} {date} {chart_link}")
                # self.records.setdefault(stockcode, {})["Date"] = date                

            return results

        length_dataframe = len(dataframe)
        logging.info(f"Total number of available stockcodes: {length_dataframe}")
        logging.info(f"Screening past {days} days transaction period.")

        self.records = mThread(stockcodes=dataframe["Code"].to_list(), results=self.records)
        # self.records = mThread(stockcodes=["5216"], results=self.records)
        logging.info(f"Number of records: {len(self.records)}")
        logging.info(self.records)
        self.save_records()

    def generate_table(self):
        dataframe = pandas.DataFrame()
        dataframe.to_excel(excel_writer=self.record_tablepath, index=False)


if __name__ == "__main__":

    # Setup logging
    logger = logging.getLogger(name=None)
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.setLevel(level=logging.INFO)

    # Create a stream handler
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(
        fmt=logging.Formatter(
            fmt="[%(thread)5s] [%(threadName)32s] [%(asctime)s] [%(levelname)8s] : %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    )
    logger.addHandler(hdlr=stream_handler)

    fiavest_xt = FiavestXT()
    fiavest_xt.get_pillars()
    # fiavest_xt.screen_periods_of_pillars()
    # fiavest_xt.screen_latest_pillars()
    # fiavest_xt.generate_table()
    
