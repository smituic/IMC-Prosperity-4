
import json
from typing import Dict, List, Optional, Tuple

from datamodel import Order, OrderDepth, Symbol, TradingState


class Trader:
    POSITION_LIMITS = {
        "ASH_COATED_OSMIUM": 80,
        "INTARIAN_PEPPER_ROOT": 80,
    }

    OSMIUM_FAIR = 10000.0

    # Keep Pepper EXACTLY the same as the stable round-1 style file.
    PEPPER_DRIFT_PER_TIMESTAMP = 0.01
    PEPPER_DAY0_ANCHOR = 12000.0

    PEPPER_STOP_BUY_GAP = 15.0
    PEPPER_REDUCE_GAP = 30.0
    PEPPER_REDUCE_SIZE = 20

    def bid(self):
        # Round 2 Market Access Fee bid.
        return 75

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

    def _best_volumes(self, order_depth: OrderDepth) -> Tuple[int, int]:
        best_bid, best_ask = self._best_bid_ask(order_depth)
        bid_vol = order_depth.buy_orders.get(best_bid, 0) if best_bid is not None else 0
        ask_vol = -order_depth.sell_orders.get(best_ask, 0) if best_ask is not None else 0
        return max(0, bid_vol), max(0, ask_vol)

    def _mid_price(self, order_depth: OrderDepth) -> Optional[float]:
        best_bid, best_ask = self._best_bid_ask(order_depth)
        if best_bid is None and best_ask is None:
            return None
        if best_bid is None:
            return float(best_ask)
        if best_ask is None:
            return float(best_bid)
        return (best_bid + best_ask) / 2.0

    def _spread(self, order_depth: OrderDepth) -> Optional[int]:
        best_bid, best_ask = self._best_bid_ask(order_depth)
        if best_bid is None or best_ask is None:
            return None
        return best_ask - best_bid

    def _microprice(self, order_depth: OrderDepth) -> Optional[float]:
        best_bid, best_ask = self._best_bid_ask(order_depth)
        if best_bid is None or best_ask is None:
            return self._mid_price(order_depth)

        bid_vol, ask_vol = self._best_volumes(order_depth)
        if bid_vol + ask_vol == 0:
            return (best_bid + best_ask) / 2.0

        return (best_ask * bid_vol + best_bid * ask_vol) / (bid_vol + ask_vol)

    def _imbalance(self, order_depth: OrderDepth) -> float:
        bid_vol, ask_vol = self._best_volumes(order_depth)
        total = bid_vol + ask_vol
        if total == 0:
            return 0.0
        return (bid_vol - ask_vol) / total

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

    # ----------------------------
    # Slight Osmium squeeze only
    # ----------------------------
    def _osmium_fair(self, order_depth: OrderDepth) -> float:
        mid = self._mid_price(order_depth)
        micro = self._microprice(order_depth)
        imbalance = self._imbalance(order_depth)

        if mid is None and micro is None:
            return self.OSMIUM_FAIR
        if mid is None:
            mid = micro
        if micro is None:
            micro = mid

        # Small upgrade only: still anchored at 10,000,
        # but use a little book pressure as well.
        return (
            0.70 * self.OSMIUM_FAIR
            + 0.15 * mid
            + 0.15 * micro
            + 0.8 * imbalance
        )

    def _osmium_zero_edge_unwind(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        position: int,
    ) -> Tuple[List[Order], int]:
        orders: List[Order] = []
        best_bid, best_ask = self._best_bid_ask(order_depth)
        if best_bid is None or best_ask is None:
            return orders, position

        # Small, conservative flattening only when inventory is stretched.
        if position >= 40 and best_bid >= fair_value - 0.5:
            qty = min(position, order_depth.buy_orders[best_bid], 10)
            if qty > 0:
                orders.append(Order(product, best_bid, -qty))
                position -= qty

        if position <= -40 and best_ask <= fair_value + 0.5:
            available = -order_depth.sell_orders[best_ask]
            qty = min(-position, available, 10)
            if qty > 0:
                orders.append(Order(product, best_ask, qty))
                position += qty

        return orders, position

    def _make_osmium_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        position: int,
    ) -> List[Order]:
        orders: List[Order] = []
        best_bid, best_ask = self._best_bid_ask(order_depth)
        if best_bid is None or best_ask is None:
            return orders

        spread = self._spread(order_depth) or 16
        skew = position / self.POSITION_LIMITS[product]
        imbalance = self._imbalance(order_depth)

        buy_capacity = self._allowable_buy(product, position)
        sell_capacity = self._allowable_sell(product, position)
        if buy_capacity <= 0 and sell_capacity <= 0:
            return orders

        # Only small spread-aware change from the stable file.
        if spread >= 18:
            buy_offset = 3
            sell_offset = 3
            base_size = 28
        elif spread >= 16:
            buy_offset = 3
            sell_offset = 3
            base_size = 25
        else:
            buy_offset = 4
            sell_offset = 4
            base_size = 20

        # Slightly stronger inventory control than the generic helper.
        inventory_shift = int(round(3 * skew))
        imbalance_shift = int(round(1.5 * imbalance))

        raw_buy = int(fair_value - buy_offset) - inventory_shift + max(0, imbalance_shift)
        raw_sell = int(fair_value + sell_offset) - inventory_shift + min(0, imbalance_shift)

        buy_quote = min(best_bid + 1, raw_buy)
        sell_quote = max(best_ask - 1, raw_sell)

        if buy_quote >= sell_quote:
            buy_quote = best_bid
            sell_quote = best_ask

        buy_size = min(
            buy_capacity,
            max(0, base_size - int(max(0.0, skew) * 8)),
        )
        sell_size = min(
            sell_capacity,
            max(0, base_size - int(max(0.0, -skew) * 8)),
        )

        if position >= 45:
            sell_quote = max(best_bid + 1, min(sell_quote, int(fair_value + 1)))
            sell_size = min(sell_capacity, sell_size + 5)
        elif position <= -45:
            buy_quote = min(best_ask - 1, max(buy_quote, int(fair_value - 1)))
            buy_size = min(buy_capacity, buy_size + 5)

        if buy_size > 0:
            orders.append(Order(product, buy_quote, buy_size))
        if sell_size > 0:
            orders.append(Order(product, sell_quote, -sell_size))

        return orders

    # ----------------------------
    # Pepper: unchanged from stable file
    # ----------------------------
    def _pepper_fair(self, state: TradingState, order_depth: OrderDepth, memory: dict) -> float:
        mid = self._mid_price(order_depth)
        if mid is None:
            anchor = memory.get("pepper_anchor", self.PEPPER_DAY0_ANCHOR)
            return anchor + self.PEPPER_DRIFT_PER_TIMESTAMP * state.timestamp

        raw_anchor = mid - self.PEPPER_DRIFT_PER_TIMESTAMP * state.timestamp
        prev_anchor = memory.get("pepper_anchor")
        if prev_anchor is None:
            anchor = raw_anchor
        else:
            anchor = 0.95 * prev_anchor + 0.05 * raw_anchor

        memory["pepper_anchor"] = anchor
        return anchor + self.PEPPER_DRIFT_PER_TIMESTAMP * state.timestamp

    def run(self, state: TradingState):
        result: Dict[Symbol, List[Order]] = {}
        memory = self._load_data(state.traderData)

        # ---- ASH_COATED_OSMIUM ----
        product = "ASH_COATED_OSMIUM"
        if product in state.order_depths:
            order_depth = state.order_depths[product]
            position = state.position.get(product, 0)
            fair = self._osmium_fair(order_depth)

            orders, post_unwind_pos = self._osmium_zero_edge_unwind(
                product=product,
                order_depth=order_depth,
                fair_value=fair,
                position=position,
            )

            spread = self._spread(order_depth) or 16
            if spread >= 18:
                buy_edge = 0.5
                sell_edge = 0.5
            elif spread >= 16:
                buy_edge = 1.0
                sell_edge = 1.0
            else:
                buy_edge = 1.5
                sell_edge = 1.5

            adjusted_fair = fair - 0.05 * position

            new_orders, post_take_pos = self._take_orders(
                product=product,
                order_depth=order_depth,
                fair_value=adjusted_fair,
                position=post_unwind_pos,
                buy_edge=buy_edge,
                sell_edge=sell_edge,
            )
            orders.extend(new_orders)

            orders.extend(
                self._make_osmium_orders(
                    product=product,
                    order_depth=order_depth,
                    fair_value=adjusted_fair,
                    position=post_take_pos,
                )
            )
            result[product] = orders

        # ---- INTARIAN_PEPPER_ROOT ----
        product = "INTARIAN_PEPPER_ROOT"
        if product in state.order_depths:
            order_depth = state.order_depths[product]
            position = state.position.get(product, 0)
            fair = self._pepper_fair(state, order_depth, memory)

            mid = self._mid_price(order_depth)
            best_bid, _ = self._best_bid_ask(order_depth)

            stop_buy = False
            reduce_long = False

            if mid is not None:
                if mid < fair - self.PEPPER_STOP_BUY_GAP:
                    stop_buy = True
                if mid < fair - self.PEPPER_REDUCE_GAP:
                    reduce_long = True

            orders: List[Order] = []

            if reduce_long and position > 0 and best_bid is not None:
                qty = min(position, self.PEPPER_REDUCE_SIZE)
                if qty > 0:
                    orders.append(Order(product, best_bid, -qty))
                    position -= qty

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
