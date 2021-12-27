import asyncio
import itertools
import time
import numpy as np
import math
from queue import Queue
from threading import Timer

from ready_trader_go import BaseAutoTrader, Instrument, Lifespan, Side

POSITION_LIMIT = 1000
TICK_SIZE_IN_CENTS = 1
ACTIVE_ORDER_COUNT_LIMIT = 10
ACTIVE_VOLUME_LIMIT = 200

RATELIMIT = 24
MIN_VOLUME_OF_INTEREST = 20
SIG_STRETCH = 100.

class AutoTrader(BaseAutoTrader):
    def __init__(self, loop: asyncio.AbstractEventLoop, team_name: str, secret: str):
        super().__init__(loop, team_name, secret)
        self.order_ids = itertools.count(1)
        self.bids = dict()
        self.asks = dict()
        self.ask_id = self.ask_price = self.bid_id = self.bid_price = self.position = 0
        self.ftrmid = -1

        self.wmp = -1
        self.ask_prices = -1
        self.ask_volumes = -1
        self.bid_prices = -1
        self.bid_volumes = -1
        self.bestBid = -1
        self.bestAsk = -1

        self.awaitingCancel = set()
        self.count = 0
        self.tick = 0

    def on_order_book_update_message(self, instrument, sequence_number, ask_prices, ask_volumes, bid_prices, bid_volumes):

        if (bid_volumes[0] == 0 and ask_volumes[0] == 0):
            return

        i = (bid_volumes[0] / (bid_volumes[0] + ask_volumes[0]))
        wmp = i*ask_prices[0] + (1-i)*bid_prices[0]

        if (instrument == 0):
            self.ftrmid = wmp
            return
        self.tick += 1
        if (self.tick % 4 == 0):
            self.count = 0

        bestAsk, bestBid = self.relevantPrices(ask_prices, ask_volumes, bid_prices, bid_volumes)
        self.wmp = wmp
        self.ask_prices = ask_prices
        self.ask_volumes = ask_volumes
        self.bid_prices = bid_prices
        self.bid_volumes = bid_volumes
        self.bestBid = bestBid
        self.bestAsk = bestAsk

        self.deleteDeadTrades(bestBid, bestAsk)
        if (self.activeVolume() < 150):
            self.makeTrade(self.wmp, self.ask_prices, self.ask_volumes, self.bid_prices, self.bid_volumes, self.bestBid, self.bestAsk)

    def makeTrade(self, midprice, ask_prices, ask_volumes, bid_prices, bid_volumes, bestBid, bestAsk):
        askV, bidV = self.manageVolumes(midprice)
        if (askV <= 0 or bidV <= 0):
            return

        askP, bidP = self.managePrices(midprice, bestBid, bestAsk)
        if (self.count < RATELIMIT-1):
            self.count += 2
            bidId, askId = next(self.order_ids), next(self.order_ids)
            self.send_insert_order(bidId, Side.BUY, bidP, bidV, Lifespan.GOOD_FOR_DAY)
            self.send_insert_order(askId, Side.SELL, askP, askV, Lifespan.GOOD_FOR_DAY)
            self.bids[bidId] = {"price": bidP, "volume": bidV}
            self.asks[askId] = {"price": askP, "volume": askV}

    def manageVolumes(self, wmp):
        if (len(self.bids) + len(self.asks) >= ACTIVE_ORDER_COUNT_LIMIT - 1):
            return (0, 0)
        if (self.activeVolume() >= ACTIVE_VOLUME_LIMIT):
            return (0, 0)

        left = ACTIVE_VOLUME_LIMIT - self.activeVolume()

        if (left < 20):
            return (0,0)

        middiff = (self.ftrmid - wmp)/20.
        asks = self.sigmoid(self.position - middiff*(POSITION_LIMIT))
        askVol = round(left*asks)
        bidVol = left-askVol
        if (self.position + bidVol >= POSITION_LIMIT):
            askVol += bidVol
            bidVol = 0
        if (self.position - askVol <= -POSITION_LIMIT):
            bidVol += askVol
            askVol = 0
        return (askVol, bidVol)

    def managePrices(self, midprice, bestBid, bestAsk):
        if (self.ftrmid >= bestBid and self.ftrmid <= bestAsk):
            askP = round(midprice + 0.6)
            bidP = round(midprice - 0.6)
        elif (self.ftrmid < bestBid):
            askP = round(midprice + 0.6)
            bidP = bestBid
        else:
            askP = bestAsk
            bidP = round(midprice - 0.6)
        self.ask_price = askP
        self.bid_price = bidP
        return (askP, bidP)

    def on_order_filled_message(self, client_order_id, price, volume) -> None:
        if client_order_id in self.bids.keys():
            self.position += volume
            self.bids[client_order_id]["volume"] -= volume
        elif client_order_id in self.asks.keys():
            self.position -= volume
            self.asks[client_order_id]["volume"] -= volume

    def on_order_status_message(self, client_order_id: int, fill_volume: int, remaining_volume: int, fees: int) -> None:
        if remaining_volume == 0:
            if (client_order_id in self.bids):
                self.bids.pop(client_order_id, None)
            elif (client_order_id in self.asks):
                self.asks.pop(client_order_id, None)
            if (client_order_id in self.awaitingCancel):
                self.awaitingCancel.remove(client_order_id)
                if len(self.awaitingCancel) == 0:
                    self.makeTrade(self.wmp, self.ask_prices, self.ask_volumes, self.bid_prices, self.bid_volumes, self.bestBid, self.bestAsk)

    def activeVolume(self):
        sum = 0
        for k in self.bids.copy():
            sum += self.bids[k]["volume"]
        for k in self.asks.copy():
            sum += self.asks[k]["volume"]
        return sum

    def relevantPrices(self, ask_prices, ask_volumes, bid_prices, bid_volumes):
        i_list = np.where(np.asarray(ask_volumes) > MIN_VOLUME_OF_INTEREST)[0]
        j_list = np.where(np.asarray(bid_volumes) > MIN_VOLUME_OF_INTEREST)[0]
        if (i_list.size == 0 or j_list.size == 0):
            return (-1,-1)
        i = i_list[0]
        j = j_list[0]
        if (i >= len(ask_prices) or j >= len(bid_prices)):
            return (-1,-1)
        return (ask_prices[i], bid_prices[j])

    def deleteDeadTrades(self, bestBid, bestAsk):
        spread = bestAsk - bestBid
        for k,v in self.bids.copy().items():
            if (bestBid - v['price'] > 0 or (v['price'] - self.ftrmid) > 2 or spread > 3):
                if (self.count < RATELIMIT):
                    self.count += 1
                    self.awaitingCancel.add(k)
                    self.send_cancel_order(k)
                else:
                    return
        for k,v in self.asks.copy().items():
            if (v['price'] - bestAsk > 0 or (self.ftrmid - v['price']) > 2 or spread > 3):
                if (self.count < RATELIMIT):
                    self.count += 1
                    self.awaitingCancel.add(k)
                    self.send_cancel_order(k)
                else:
                    return

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x/SIG_STRETCH))

    def on_error_message(self, client_order_id: int, error_message: bytes):
        print('Q', error_message)
