import json
from typing import Any, Dict, List, Optional, Tuple

from datamodel import Order, OrderDepth, Symbol, TradingState


# This is our robot trader! It buys and sells things to make money.
# Think of it like a kid at a toy swap meet — buy cheap, sell expensive!
class Trader:

    # These are the max number of each toy we're allowed to hold at once.
    # The rule says: don't hold more than 80 of either thing!
    POSITION_LIMITS = {
        "ASH_COATED_OSMIUM": 80,
        "INTARIAN_PEPPER_ROOT": 80,
    }

    # Osmium (the shiny rock) is always worth around 10,000 coins.
    # It's like a toy that always costs about the same — never goes crazy high or low.
    OSMIUM_FAIR = 10000.0

    # CRITICAL FIX: drift is ~1000 price units per 100K timestamps = 0.01 per timestamp
    # The original code had 0.001, which was 10x too small, causing terrible pepper performance
    # Pepper gets more expensive by a tiny bit every single moment (like a rare candy going up in price).
    # Every tick (moment in time), pepper's fair price goes up by 0.01 coins.
    PEPPER_DRIFT_PER_TIMESTAMP = 0.01

    # Pepper anchor: calibrated from historical data
    # Day -2 started ~10000, Day -1 ~11000, Day 0 ~12000 (each day +1000)
    # On Day 0, pepper starts at 12,000 coins. Each day it goes up by 1,000 more!
    PEPPER_DAY0_ANCHOR = 12000.0

    # Safety rules for pepper — like "stop buying candy if the price looks too weird":
    PEPPER_STOP_BUY_GAP = 15.0      # If pepper price drops 15 below what we expect, stop buying more
    PEPPER_REDUCE_GAP = 30.0        # If pepper price drops 30 below what we expect, start selling some
    PEPPER_REDUCE_SIZE = 20         # When selling off, sell at most 20 at a time

    def _load_data(self, trader_data: str) -> dict:
        # Wake up our robot's memory from last time.
        # It's like reading the notes you wrote before going to sleep!
        if not trader_data:
            return {}
        try:
            return json.loads(trader_data)
        except Exception:
            return {}

    def _dump_data(self, data: dict) -> str:
        # Save our robot's memory so it remembers next time it wakes up.
        # Like writing notes before going to sleep!
        try:
            return json.dumps(data)
        except Exception:
            return ""

    def _best_bid_ask(self, order_depth: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
        # Find the best prices right now in the market.
        # best_bid = highest price someone is willing to PAY (best buyer)
        # best_ask = lowest price someone is willing to SELL for (cheapest seller)
        best_bid = max(order_depth.buy_orders) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders) if order_depth.sell_orders else None
        return best_bid, best_ask

    def _mid_price(self, order_depth: OrderDepth) -> Optional[float]:
        # The "middle price" — halfway between the buyer's price and the seller's price.
        # Like if someone wants to buy for $8 and someone wants to sell for $10,
        # the mid price is $9 — right in the middle!
        best_bid, best_ask = self._best_bid_ask(order_depth)
        if best_bid is None and best_ask is None:
            return None
        if best_bid is None:
            return float(best_ask)
        if best_ask is None:
            return float(best_bid)
        return (best_bid + best_ask) / 2.0

    def _allowable_buy(self, product: str, position: int) -> int:
        # How many more can we BUY without breaking the "don't hold too many" rule?
        # Like checking how many more cookies you're allowed to eat before dinner!
        return max(0, self.POSITION_LIMITS[product] - position)

    def _allowable_sell(self, product: str, position: int) -> int:
        # How many can we SELL without going negative (you can't sell what you don't have!)?
        # Like checking you actually have cookies before giving them away.
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
        # This function JUMPS at good deals already in the market.
        # If someone is selling too cheap → we buy it fast!
        # If someone is paying too much → we sell to them fast!
        # Think of it like grabbing candy on sale before anyone else does.
        orders: List[Order] = []

        # --- BUY SIDE: grab cheap sellers ---
        buy_remaining = self._allowable_buy(product, position)
        for ask_price in sorted(order_depth.sell_orders):
            # Only buy if the price is cheap enough (below fair minus our required discount)
            if ask_price > fair_value - buy_edge or buy_remaining <= 0:
                break
            available = -order_depth.sell_orders[ask_price]
            qty = min(available, buy_remaining)  # Don't buy more than we're allowed
            if qty > 0:
                orders.append(Order(product, ask_price, qty))
                position += qty
                buy_remaining -= qty

        # --- SELL SIDE: sell to buyers paying too much ---
        sell_remaining = self._allowable_sell(product, position)
        for bid_price in sorted(order_depth.buy_orders, reverse=True):
            # Only sell if the buyer is paying enough (above fair plus our required bonus)
            if bid_price < fair_value + sell_edge or sell_remaining <= 0:
                break
            available = order_depth.buy_orders[bid_price]
            qty = min(available, sell_remaining)  # Don't sell more than we have
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
        # This function puts up OUR OWN price signs in the market — like a shop stall.
        # We say: "I'll buy for THIS price" and "I'll sell for THAT price."
        # Then we just wait for someone to accept our offer.
        orders: List[Order] = []
        best_bid, best_ask = self._best_bid_ask(order_depth)
        if best_bid is None or best_ask is None:
            return orders  # Nobody is trading right now, go home

        buy_capacity = self._allowable_buy(product, position)
        sell_capacity = self._allowable_sell(product, position)
        if buy_capacity <= 0 and sell_capacity <= 0:
            return orders  # We're completely full OR completely empty, nothing to do

        # Inventory skew: if we're holding a LOT, nudge prices down a little to sell more.
        # If we're holding a LITTLE, nudge prices up to buy more.
        # Like a store having a sale when they have too much stock!
        skew = position / self.POSITION_LIMITS[product]
        inventory_shift = int(round(2 * skew))

        # Calculate the prices we want to post for buying and selling
        raw_buy = int(fair_value - buy_offset) - inventory_shift
        raw_sell = int(fair_value + sell_offset) - inventory_shift

        # Make sure our buy price is just above the current best buyer (to be first in line!)
        buy_quote = min(best_bid + 1, raw_buy)
        # Make sure our sell price is just below the current cheapest seller (to be first in line!)
        sell_quote = max(best_ask - 1, raw_sell)

        # Safety check: our buy price should never be higher than our sell price — that's losing money!
        if buy_quote >= sell_quote:
            buy_quote = best_bid
            sell_quote = best_ask

        # How much to buy/sell: reduce size if we already hold a lot (don't go overboard)
        buy_size = min(buy_capacity, max(0, int(base_size * (1.0 - max(0.0, skew)))))
        sell_size = min(sell_capacity, max(0, int(base_size * (1.0 + min(0.0, skew)))))

        if buy_size > 0:
            orders.append(Order(product, buy_quote, buy_size))   # Post our "I want to BUY" sign
        if sell_size > 0 and not buy_only:
            orders.append(Order(product, sell_quote, -sell_size))  # Post our "I want to SELL" sign

        return orders

    def _osmium_fair(self, order_depth: OrderDepth) -> float:
        # Figure out how much Osmium is REALLY worth right now.
        # Osmium is like a very stable toy — it's almost always worth 10,000.
        # We mostly trust 10,000 (75%), but also peek at the current market price (25%).
        mid = self._mid_price(order_depth)
        if mid is None:
            return self.OSMIUM_FAIR
        # Osmium is stationary around 10,000 — anchor strongly to that level
        # but allow ~25% weight on current mid to handle any small drift
        return 0.75 * self.OSMIUM_FAIR + 0.25 * mid

    def _pepper_fair(self, state: TradingState, order_depth: OrderDepth, memory: dict) -> float:
        # Figure out how much Pepper is REALLY worth right now.
        # Pepper is like a toy that gets more and more expensive every single second!
        # We use our memory (saved notes) to track where the price started, then add the drift.
        mid = self._mid_price(order_depth)
        if mid is None:
            # No market data — just use our saved starting price + time drift
            anchor = memory.get("pepper_anchor", self.PEPPER_DAY0_ANCHOR)
            return anchor + self.PEPPER_DRIFT_PER_TIMESTAMP * state.timestamp

        # Work backwards: if we know the price now and how fast it drifts,
        # we can figure out what it was at time zero (the "anchor").
        raw_anchor = mid - self.PEPPER_DRIFT_PER_TIMESTAMP * state.timestamp
        prev_anchor = memory.get("pepper_anchor")
        if prev_anchor is None:
            anchor = raw_anchor  # First time — just trust the current observation
        else:
            # Slow update: adapts to any offset between days, but resists short-term noise
            # Mix mostly old anchor (95%) with small update from new data (5%) — very stable!
            anchor = 0.95 * prev_anchor + 0.05 * raw_anchor

        memory["pepper_anchor"] = anchor  # Save the anchor for next time
        return anchor + self.PEPPER_DRIFT_PER_TIMESTAMP * state.timestamp

    def run(self, state: TradingState):
        # This is the MAIN function — it runs every single moment of the game.
        # It looks at the market, decides what to buy or sell, and sends those orders.
        result: Dict[Symbol, List[Order]] = {}
        memory = self._load_data(state.traderData)  # Wake up our memory from last tick

        # ── ASH_COATED_OSMIUM: stationary mean-reversion market making ─────────
        # OSMIUM STRATEGY: This toy always goes back to its normal price (~10,000).
        # So we buy when it's a little cheap, sell when it's a little expensive.
        # Like buying lemonade when it's on sale and selling when the price goes back up!
        product = "ASH_COATED_OSMIUM"
        if product in state.order_depths:
            order_depth = state.order_depths[product]
            position = state.position.get(product, 0)  # How many we currently hold
            fair = self._osmium_fair(order_depth)       # What we think it's worth

            # Slightly tighter edge vs original (was 2/2) to cross more spread-widenings
            # Step 1: Grab any existing cheap sellers or expensive buyers immediately
            orders, post_take_pos = self._take_orders(
                product=product,
                order_depth=order_depth,
                fair_value=fair,
                position=position,
                buy_edge=1,    # buy when ask <= fair - 1
                sell_edge=1,   # sell when bid >= fair + 1
            )
            # Step 2: Put up our own buy/sell signs at nice prices and wait
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
        # PEPPER STRATEGY: This toy gets MORE EXPENSIVE every single day (+1000 per day!).
        # So the best plan is: BUY AS MUCH AS POSSIBLE and hold on tight!
        # We almost NEVER sell — we just keep accumulating like collecting rare stickers.
        # The longer we hold, the more money we make as the price keeps climbing!
        product = "INTARIAN_PEPPER_ROOT"
        if product in state.order_depths:
            order_depth = state.order_depths[product]
            position = state.position.get(product, 0)  # How many pepper we currently hold
            fair = self._pepper_fair(state, order_depth, memory)  # Expected pepper price right now

            mid = self._mid_price(order_depth)
            best_bid, best_ask = self._best_bid_ask(order_depth)

            # Safety flags — like warning lights on a dashboard
            stop_buy = False   # Should we pause buying?
            reduce_long = True  # Should we sell some off?
            reduce_long = False

            if mid is not None:
                # If the real price is WAY below what we expect, something is weird — be careful!
                if mid < fair - self.PEPPER_STOP_BUY_GAP:
                    stop_buy = True   # Price is 15+ below expected → pause buying for now
                if mid < fair - self.PEPPER_REDUCE_GAP:
                    reduce_long = True  # Price is 30+ below expected → sell some to be safe

            # buy_edge = -10: buy even if ask is up to 10 above fair value
            #   → aggressively crosses any asks within 10 ticks of the trending fair price
            # sell_edge = 9999: only sell if bid is 9999 above fair (= never sell by crossing)

            # buy_offset = -5 → passive bid at fair + 5, aggressive
            # sell_offset = 200 → passive ask 200 above fair, won't get filled
            # buy_only = True → skip passive sell quotes entirely
            orders: List[Order] = []

            # Harder regime break: reduce some Pepper if price is way below our trend/fair
            # Emergency sell! If price crashed way below expectations, dump a few to protect profits.
            # Like selling your stickers quick if you hear their value might drop even more.
            if reduce_long and position > 0 and best_bid is not None:
                qty = min(position, self.PEPPER_REDUCE_SIZE)  # Sell at most 20 at a time
                if qty > 0:
                    orders.append(Order(product, best_bid, -qty))  # Sell now at the best available price
                    position -= qty

            # Normal Pepper behavior only if we are NOT in stop-buy mode
            # If everything looks normal, go full aggressive buying mode!
            if not stop_buy:
                # Grab any pepper being sold near our expected fair price — don't hesitate!
                new_orders, post_take_pos = self._take_orders(
                    product=product,
                    order_depth=order_depth,
                    fair_value=fair,
                    position=position,
                    buy_edge=-10,   # Buy even if up to 10 coins ABOVE our fair value (very aggressive!)
                    sell_edge=9999, # Almost never sell through this route (9999 = basically never)
                )
                orders.extend(new_orders)

                # Also put up our own buy sign at a great price — just in case someone wants to sell cheap
                orders.extend(
                    self._make_orders(
                        product=product,
                        order_depth=order_depth,
                        fair_value=fair,
                        position=post_take_pos,
                        buy_offset=-5,   # Post buy offer at fair + 5 (paying a bit above fair to get filled!)
                        sell_offset=200, # Post sell offer 200 above fair — so high nobody will buy it
                        base_size=40,    # Try to buy up to 40 at a time — go big!
                        buy_only=True,   # Only put up BUY signs, never SELL signs for pepper
                    )
                )

            result[product] = orders

        # Save our memory so the robot remembers everything next time it wakes up
        trader_data = self._dump_data(memory)
        conversions = 0  # No currency conversions needed this round
        return result, conversions, trader_data
