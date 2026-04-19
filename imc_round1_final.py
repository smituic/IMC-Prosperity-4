import json
from typing import Any, Dict, List, Optional, Tuple

from datamodel import Order, OrderDepth, Symbol, TradingState


class Trader:
    POSITION_LIMITS = {
        "ASH_COATED_OSMIUM": 80,
        "INTARIAN_PEPPER_ROOT": 80,
    }

    OSMIUM_FAIR = 10000.0

    # CRITICAL FIX: drift is ~1000 price units per 100K timestamps = 0.01 per timestamp
    # The original code had 0.001, which was 10x too small, causing terrible pepper performance
    PEPPER_DRIFT_PER_TIMESTAMP = 0.01

    # Pepper anchor: calibrated from historical data
    # Day -2 started ~10000, Day -1 ~11000, Day 0 ~12000 (each day +1000)
    PEPPER_DAY0_ANCHOR = 12000.0

    PEPPER_STOP_BUY_GAP = 15.0      # if mid < fair - 15, stop adding new longs
    PEPPER_REDUCE_GAP = 30.0        # if mid < fair - 30, reduce existing long
    PEPPER_REDUCE_SIZE = 20         # sell at most 20 lots when hard stop triggers

    def _load_data(self, trader_data: str) -> dict:
        if not trader_data:
            return {}
        try:
            return json.loads(trader_data)
        except Exception:
            return {}

    def _dump_data(self, data: dict) -> str:
        try:
            return json.dumps(data)
        except Exception:
            return ""

    def _best_bid_ask(self, order_depth: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
        best_bid = max(order_depth.buy_orders) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders) if order_depth.sell_orders else None
        return best_bid, best_ask

    def _mid_price(self, order_depth: OrderDepth) -> Optional[float]:
        best_bid, best_ask = self._best_bid_ask(order_depth)
        if best_bid is None and best_ask is None:
            return None
        if best_bid is None:
            return float(best_ask)
        if best_ask is None:
            return float(best_bid)
        return (best_bid + best_ask) / 2.0

    def _allowable_buy(self, product: str, position: int) -> int:
        return max(0, self.POSITION_LIMITS[product] - position)

    def _allowable_sell(self, product: str, position: int) -> int:
        return max(0, self.POSITION_LIMITS[product] + position)

    def _take_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        position: int,
        buy_edge: float,
        sell_edge: float,
    ) -> Tuple[List[Order], int]:
        """Cross the spread when edge condition is met.
        
        Buys when ask_price <= fair_value - buy_edge
        Sells when bid_price >= fair_value + sell_edge
        
        Negative buy_edge = willing to buy ABOVE fair value (aggressive)
        Large sell_edge = almost never sell by crossing
        """
        orders: List[Order] = []

        buy_remaining = self._allowable_buy(product, position)
        for ask_price in sorted(order_depth.sell_orders):
            if ask_price > fair_value - buy_edge or buy_remaining <= 0:
                break
            available = -order_depth.sell_orders[ask_price]
            qty = min(available, buy_remaining)
            if qty > 0:
                orders.append(Order(product, ask_price, qty))
                position += qty
                buy_remaining -= qty

        sell_remaining = self._allowable_sell(product, position)
        for bid_price in sorted(order_depth.buy_orders, reverse=True):
            if bid_price < fair_value + sell_edge or sell_remaining <= 0:
                break
            available = order_depth.buy_orders[bid_price]
            qty = min(available, sell_remaining)
            if qty > 0:
                orders.append(Order(product, bid_price, -qty))
                position -= qty
                sell_remaining -= qty

        return orders, position

    def _make_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        position: int,
        buy_offset: int,
        sell_offset: int,
        base_size: int,
        buy_only: bool = False,
    ) -> List[Order]:
        """Post passive quotes with inventory skew.
        
        buy_offset:  buy quote placed at fair - buy_offset (negative = above fair)
        sell_offset: sell quote placed at fair + sell_offset (large = far above fair)
        buy_only:    if True, skip the passive sell quote
        """
        orders: List[Order] = []
        best_bid, best_ask = self._best_bid_ask(order_depth)
        if best_bid is None or best_ask is None:
            return orders

        buy_capacity = self._allowable_buy(product, position)
        sell_capacity = self._allowable_sell(product, position)
        if buy_capacity <= 0 and sell_capacity <= 0:
            return orders

        skew = position / self.POSITION_LIMITS[product]
        inventory_shift = int(round(2 * skew))

        raw_buy = int(fair_value - buy_offset) - inventory_shift
        raw_sell = int(fair_value + sell_offset) - inventory_shift

        buy_quote = min(best_bid + 1, raw_buy)
        sell_quote = max(best_ask - 1, raw_sell)

        if buy_quote >= sell_quote:
            buy_quote = best_bid
            sell_quote = best_ask

        buy_size = min(buy_capacity, max(0, int(base_size * (1.0 - max(0.0, skew)))))
        sell_size = min(sell_capacity, max(0, int(base_size * (1.0 + min(0.0, skew)))))

        if buy_size > 0:
            orders.append(Order(product, buy_quote, buy_size))
        if sell_size > 0 and not buy_only:
            orders.append(Order(product, sell_quote, -sell_size))

        return orders

    def _osmium_fair(self, order_depth: OrderDepth) -> float:
        mid = self._mid_price(order_depth)
        if mid is None:
            return self.OSMIUM_FAIR
        # Osmium is stationary around 10,000 — anchor strongly to that level
        # but allow ~25% weight on current mid to handle any small drift
        return 0.75 * self.OSMIUM_FAIR + 0.25 * mid

    def _pepper_fair(self, state: TradingState, order_depth: OrderDepth, memory: dict) -> float:
        mid = self._mid_price(order_depth)
        if mid is None:
            anchor = memory.get("pepper_anchor", self.PEPPER_DAY0_ANCHOR)
            return anchor + self.PEPPER_DRIFT_PER_TIMESTAMP * state.timestamp

        # Back out the implied anchor at t=0 from current mid
        raw_anchor = mid - self.PEPPER_DRIFT_PER_TIMESTAMP * state.timestamp
        prev_anchor = memory.get("pepper_anchor")
        if prev_anchor is None:
            anchor = raw_anchor
        else:
            # Slow update: adapts to any offset between days, but resists short-term noise
            anchor = 0.95 * prev_anchor + 0.05 * raw_anchor

        memory["pepper_anchor"] = anchor
        return anchor + self.PEPPER_DRIFT_PER_TIMESTAMP * state.timestamp

    def run(self, state: TradingState):
        result: Dict[Symbol, List[Order]] = {}
        memory = self._load_data(state.traderData)

        # ── ASH_COATED_OSMIUM: stationary mean-reversion market making ─────────
        product = "ASH_COATED_OSMIUM"
        if product in state.order_depths:
            order_depth = state.order_depths[product]
            position = state.position.get(product, 0)
            fair = self._osmium_fair(order_depth)

            # Slightly tighter edge vs original (was 2/2) to cross more spread-widenings
            orders, post_take_pos = self._take_orders(
                product=product,
                order_depth=order_depth,
                fair_value=fair,
                position=position,
                buy_edge=1,    # buy when ask <= fair - 1
                sell_edge=1,   # sell when bid >= fair + 1
            )
            orders.extend(
                self._make_orders(
                    product=product,
                    order_depth=order_depth,
                    fair_value=fair,
                    position=post_take_pos,
                    buy_offset=3,    # passive bid at fair - 3 (tighter than original 4)
                    sell_offset=3,   # passive ask at fair + 3
                    base_size=25,    # bigger than original 16 for more volume
                )
            )
            result[product] = orders

        # ── INTARIAN_PEPPER_ROOT: strong upward trend — stay MAX LONG ──────────
        #
        # With drift = +1000 per day and position limit = 80:
        #   Max theoretical profit = 80 × 1000 = 80,000
        #
        # Strategy: aggressively accumulate to max long, almost never sell.
        # The carry (holding the trend) dominates any spread-capture logic.
        product = "INTARIAN_PEPPER_ROOT"
        if product in state.order_depths:
            order_depth = state.order_depths[product]
            position = state.position.get(product, 0)
            fair = self._pepper_fair(state, order_depth, memory)

            mid = self._mid_price(order_depth)
            best_bid, best_ask = self._best_bid_ask(order_depth)

            stop_buy = False
            reduce_long = False

            if mid is not None:
                if mid < fair - self.PEPPER_STOP_BUY_GAP:
                    stop_buy = True
                if mid < fair - self.PEPPER_REDUCE_GAP:
                    reduce_long = True



            # buy_edge = -10: buy even if ask is up to 10 above fair value
            #   → aggressively crosses any asks within 10 ticks of the trending fair price
            # sell_edge = 9999: only sell if bid is 9999 above fair (= never sell by crossing)
            
            # buy_offset = -5 → passive bid at fair + 5, aggressive
            # sell_offset = 200 → passive ask 200 above fair, won't get filled
            # buy_only = True → skip passive sell quotes entirely
            orders: List[Order] = []

            # Harder regime break: reduce some Pepper if price is way below our trend/fair
            if reduce_long and position > 0 and best_bid is not None:
                qty = min(position, self.PEPPER_REDUCE_SIZE)
                if qty > 0:
                    orders.append(Order(product, best_bid, -qty))
                    position -= qty

            # Normal Pepper behavior only if we are NOT in stop-buy mode
            if not stop_buy:
                new_orders, post_take_pos = self._take_orders(
                    product=product,
                    order_depth=order_depth,
                    fair_value=fair,
                    position=position,
                    buy_edge=-10,
                    sell_edge=9999,
                )
                orders.extend(new_orders)

                orders.extend(
                    self._make_orders(
                        product=product,
                        order_depth=order_depth,
                        fair_value=fair,
                        position=post_take_pos,
                        buy_offset=-5,
                        sell_offset=200,
                        base_size=40,
                        buy_only=True,
                    )
                )

            result[product] = orders

        trader_data = self._dump_data(memory)
        conversions = 0
        return result, conversions, trader_data

    
