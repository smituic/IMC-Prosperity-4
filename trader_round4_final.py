from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Tuple
import math
import json


class Trader:
    """
    Round 4 trader:
      - Market-makes HYDROGEL_PACK around a mean-reverting fair value.
      - Market-makes VELVETFRUIT_EXTRACT around an adaptive EMA fair value.
      - Trades VEV_* vouchers with a Black-Scholes + implied-volatility smile model.

    This file is intentionally stdlib-only: no numpy/scipy/pandas needed in the game runner.
    """

    HYDROGEL = "HYDROGEL_PACK"
    VELVET = "VELVETFRUIT_EXTRACT"

    STRIKES: Dict[str, int] = {
        "VEV_4000": 4000,
        "VEV_4500": 4500,
        "VEV_5000": 5000,
        "VEV_5100": 5100,
        "VEV_5200": 5200,
        "VEV_5300": 5300,
        "VEV_5400": 5400,
        "VEV_5500": 5500,
        "VEV_6000": 6000,
        "VEV_6500": 6500,
    }

    # Backtest filter from the run you shared:
    # keep only voucher products that were positive across the 3 historical days.
    ACTIVE_VOUCHERS = {"VEV_4000", "VEV_4500", "VEV_5100", "VEV_5200", "VEV_5300", "VEV_5500"}

    LIMITS: Dict[str, int] = {
        HYDROGEL: 200,
        VELVET: 200,
        "VEV_4000": 300,
        "VEV_4500": 300,
        "VEV_5000": 300,
        "VEV_5100": 300,
        "VEV_5200": 300,
        "VEV_5300": 300,
        "VEV_5400": 300,
        "VEV_5500": 300,
        "VEV_6000": 300,
        "VEV_6500": 300,
    }

    # Round 4 final simulation starts at 4 days to expiry.
    TTE_DAYS = 4.0
    TTE_YEARS = TTE_DAYS / 365.0

    # Conservative position sizes. Raise carefully only after backtesting.
    DELTA_MM_SIZE = {
        HYDROGEL: 35,
        VELVET: 45,
    }
    DELTA_TAKE_SIZE = {
        HYDROGEL: 45,
        VELVET: 55,
    }
    OPTION_TAKE_SIZE = 60
    OPTION_MM_SIZE = 45

    # Round 4 counterparty signals, estimated from the Data Capsule.
    # Positive weight means "follow this Mark": if they buy, fair goes up; if they sell, fair goes down.
    # Negative weight means "fade this Mark": if they buy, fair goes down; if they sell, fair goes up.
    MARK_WEIGHTS: Dict[str, Dict[str, float]] = {
        # Round 4 first-window analysis: Mark 38 is the short-term informed Hydrogel trader;
        # Mark 14/22 tend to be fade signals at the 500-3000 timestamp horizon.
        HYDROGEL: {
            "Mark 38": 0.90,
            "Mark 14": -0.70,
            "Mark 22": -1.10,
        },
        VELVET: {},
        "VEV_5300": {},
        "VEV_5400": {},
        "VEV_5500": {},
    }

    MARK_SCALE: Dict[str, float] = {
        HYDROGEL: 0.22,
        VELVET: 0.18,
        "VEV_5300": 0.20,
        "VEV_5400": 0.22,
        "VEV_5500": 0.18,
    }

    MARK_CAP: Dict[str, float] = {
        HYDROGEL: 22.0,
        VELVET: 26.0,
        "VEV_5300": 10.0,
        "VEV_5400": 10.0,
        "VEV_5500": 8.0,
    }

    def run(self, state: TradingState):
        data = self.load_data(state.traderData)
        orders: Dict[str, List[Order]] = {}

        mids = {}
        for product, depth in state.order_depths.items():
            mid = self.mid_price(depth)
            if mid is not None:
                mids[product] = mid
                self.update_ema(data, product, mid, alpha=0.08)
                self.update_ema(data, product + "__slow", mid, alpha=0.010)

        self.update_mark_signals(data, state)

        # v3: use only corrected Hydrogel Mark overlay; keep Velvet disabled.: do NOT use Mark overlay as a price signal yet. The first Mark model
        # overfit and turned Velvet into the main loser on the upload.

        # 1) HYDROGEL_PACK: keep the old mean-reversion engine.
        if self.HYDROGEL in state.order_depths:
            ema = data.get("ema", {}).get(self.HYDROGEL, mids.get(self.HYDROGEL, 9990.0))
            fair = 0.65 * ema + 0.35 * 9990.0 + self.mark_adjust(data, self.HYDROGEL)
            self.trade_delta_product(
                product=self.HYDROGEL,
                state=state,
                orders=orders,
                raw_fair=fair,
                take_edge=4.0,
                mm_width=5.0,
                pos_skew=0.055,
                max_take=self.DELTA_TAKE_SIZE[self.HYDROGEL],
                mm_size=self.DELTA_MM_SIZE[self.HYDROGEL],
            )

        # 2) VELVETFRUIT_EXTRACT standalone trading disabled. It was the upload killer.
        # We still use its mid as the underlying for VEV pricing.

        # 3) Voucher option trading.
        if self.VELVET in mids:
            spot = mids[self.VELVET]
            smile_points = self.collect_smile_points(state, spot)
            for product, strike in self.STRIKES.items():
                if product not in self.ACTIVE_VOUCHERS:
                    continue
                if product not in state.order_depths:
                    continue

                depth = state.order_depths[product]
                position = state.position.get(product, 0)

                # Deep ITM vouchers are basically spot - strike.
                # The other vouchers are valued with BS and a fitted IV smile.
                if product in ("VEV_4000", "VEV_4500"):
                    model_fair = max(0.0, spot - strike)
                    take_edge = 3.0
                    mm_width = 5.0 if product == "VEV_4000" else 4.0
                elif product in ("VEV_6000", "VEV_6500"):
                    # These usually sit at 0/1. Do not buy too much premium;
                    # only sell obvious overpricing / quote very lightly.
                    model_fair = self.option_fair_value(product, strike, spot, smile_points)
                    take_edge = 1.3
                    mm_width = 1.0
                else:
                    model_fair = self.option_fair_value(product, strike, spot, smile_points)
                    take_edge = 1.2
                    mm_width = 1.0

                # Inventory skew. If long, lower our fair; if short, raise it.
                # This stops the bot from collecting too much one-way option exposure.
                fair = model_fair - 0.025 * position + self.mark_adjust(data, product)

                # Controlled Round 4 adjustment:
                # VEV_5300 is the only voucher that has shown meaningful positive
                # contribution in both the public upload and historical checks.
                # This is not timestamp-based, so it is much less overfit than the
                # public_timed bot. If it fails, fall back to v6_best_current.
                opt_take = self.OPTION_TAKE_SIZE
                opt_mm = self.OPTION_MM_SIZE
                if product == "VEV_5300":
                    fair -= 0.75
                    opt_take = 85
                    opt_mm = 65
                elif product == "VEV_5500":
                    fair -= 0.25
                    opt_take = 65
                    opt_mm = 50

                self.trade_option_product(
                    product=product,
                    state=state,
                    orders=orders,
                    fair=fair,
                    take_edge=take_edge,
                    mm_width=mm_width,
                    max_take=opt_take,
                    mm_size=opt_mm,
                )

        trader_data = self.dump_data(data)
        conversions = 0
        return orders, conversions, trader_data

    # ---------------------------------------------------------------------
    # Generic order helpers
    # ---------------------------------------------------------------------

    def load_data(self, trader_data: str) -> dict:
        if not trader_data:
            return {"ema": {}, "marksig": {}}
        try:
            data = json.loads(trader_data)
            if "ema" not in data:
                data["ema"] = {}
            if "marksig" not in data:
                data["marksig"] = {}
            return data
        except Exception:
            return {"ema": {}, "marksig": {}}

    def dump_data(self, data: dict) -> str:
        # Keep traderData tiny.
        if "ema" in data:
            data["ema"] = {k: round(float(v), 4) for k, v in data["ema"].items()}
        if "marksig" in data:
            data["marksig"] = {k: round(float(v), 4) for k, v in data["marksig"].items()}
        return json.dumps(data, separators=(",", ":"))

    def update_mark_signals(self, data: dict, state: TradingState) -> None:
        if "marksig" not in data:
            data["marksig"] = {}

        for product, weights in self.MARK_WEIGHTS.items():
            old_signal = float(data["marksig"].get(product, 0.0))
            raw = 0.0

            for trade in state.market_trades.get(product, []):
                qty = abs(int(trade.quantity))
                buyer = getattr(trade, "buyer", None)
                seller = getattr(trade, "seller", None)
                raw += weights.get(buyer, 0.0) * qty
                raw -= weights.get(seller, 0.0) * qty

            # Delta products have longer-lasting Mark signal; options are shorter/noisier.
            decay = 0.84 if product in (self.HYDROGEL, self.VELVET) else 0.72
            val = decay * old_signal + raw
            cap = self.MARK_CAP.get(product, 10.0)
            if val > cap:
                val = cap
            elif val < -cap:
                val = -cap
            data["marksig"][product] = val

    def mark_adjust(self, data: dict, product: str) -> float:
        return float(data.get("marksig", {}).get(product, 0.0)) * self.MARK_SCALE.get(product, 0.0)

    def update_ema(self, data: dict, product: str, price: float, alpha: float = 0.08) -> None:
        if "ema" not in data:
            data["ema"] = {}
        old = data["ema"].get(product)
        if old is None:
            data["ema"][product] = float(price)
        else:
            data["ema"][product] = float(alpha * price + (1.0 - alpha) * old)

    def mid_price(self, depth: OrderDepth):
        if len(depth.buy_orders) == 0 or len(depth.sell_orders) == 0:
            return None
        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())
        return (best_bid + best_ask) / 2.0

    def best_bid_ask(self, depth: OrderDepth):
        best_bid = max(depth.buy_orders.keys()) if depth.buy_orders else None
        best_ask = min(depth.sell_orders.keys()) if depth.sell_orders else None
        return best_bid, best_ask

    def outstanding_side_qty(self, product: str, orders: Dict[str, List[Order]]) -> Tuple[int, int]:
        buy_qty = 0
        sell_qty = 0
        for order in orders.get(product, []):
            if order.quantity > 0:
                buy_qty += order.quantity
            elif order.quantity < 0:
                sell_qty += -order.quantity
        return buy_qty, sell_qty

    def buy_capacity(self, product: str, state: TradingState, orders: Dict[str, List[Order]]) -> int:
        limit = self.LIMITS[product]
        pos = state.position.get(product, 0)
        buy_used, _ = self.outstanding_side_qty(product, orders)
        return max(0, limit - pos - buy_used)

    def sell_capacity(self, product: str, state: TradingState, orders: Dict[str, List[Order]]) -> int:
        limit = self.LIMITS[product]
        pos = state.position.get(product, 0)
        _, sell_used = self.outstanding_side_qty(product, orders)
        return max(0, limit + pos - sell_used)

    def add_order(
        self,
        orders: Dict[str, List[Order]],
        product: str,
        price: int,
        quantity: int,
    ) -> None:
        if quantity == 0:
            return
        if product not in orders:
            orders[product] = []
        orders[product].append(Order(product, int(price), int(quantity)))

    # ---------------------------------------------------------------------
    # Delta-one products
    # ---------------------------------------------------------------------

    def trade_delta_product(
        self,
        product: str,
        state: TradingState,
        orders: Dict[str, List[Order]],
        raw_fair: float,
        take_edge: float,
        mm_width: float,
        pos_skew: float,
        max_take: int,
        mm_size: int,
    ) -> None:
        depth = state.order_depths[product]
        pos = state.position.get(product, 0)
        fair = raw_fair - pos_skew * pos

        # Take cheap asks.
        for ask_price, ask_volume in sorted(depth.sell_orders.items()):
            if ask_price <= fair - take_edge:
                cap = self.buy_capacity(product, state, orders)
                qty = min(cap, -ask_volume, max_take)
                if qty > 0:
                    self.add_order(orders, product, ask_price, qty)

        # Hit rich bids.
        for bid_price, bid_volume in sorted(depth.buy_orders.items(), reverse=True):
            if bid_price >= fair + take_edge:
                cap = self.sell_capacity(product, state, orders)
                qty = min(cap, bid_volume, max_take)
                if qty > 0:
                    self.add_order(orders, product, bid_price, -qty)

        best_bid, best_ask = self.best_bid_ask(depth)

        # Passive market making inside the spread.
        if best_bid is not None and best_ask is not None:
            bid = int(math.floor(fair - mm_width))
            ask = int(math.ceil(fair + mm_width))

            # Improve the book without crossing it.
            bid = max(bid, best_bid + 1)
            bid = min(bid, best_ask - 1)

            ask = min(ask, best_ask - 1)
            ask = max(ask, best_bid + 1)

            if bid < best_ask:
                qty = min(mm_size, self.buy_capacity(product, state, orders))
                if qty > 0:
                    self.add_order(orders, product, bid, qty)

            if ask > best_bid:
                qty = min(mm_size, self.sell_capacity(product, state, orders))
                if qty > 0:
                    self.add_order(orders, product, ask, -qty)

    # ---------------------------------------------------------------------
    # Options / vouchers
    # ---------------------------------------------------------------------

    def trade_option_product(
        self,
        product: str,
        state: TradingState,
        orders: Dict[str, List[Order]],
        fair: float,
        take_edge: float,
        mm_width: float,
        max_take: int,
        mm_size: int,
    ) -> None:
        depth = state.order_depths[product]

        # Take asks below fair.
        for ask_price, ask_volume in sorted(depth.sell_orders.items()):
            if ask_price <= fair - take_edge:
                cap = self.buy_capacity(product, state, orders)
                qty = min(cap, -ask_volume, max_take)
                if qty > 0:
                    self.add_order(orders, product, ask_price, qty)

        # Sell bids above fair.
        for bid_price, bid_volume in sorted(depth.buy_orders.items(), reverse=True):
            if bid_price >= fair + take_edge:
                cap = self.sell_capacity(product, state, orders)
                qty = min(cap, bid_volume, max_take)
                if qty > 0:
                    self.add_order(orders, product, bid_price, -qty)

        best_bid, best_ask = self.best_bid_ask(depth)
        if best_bid is None or best_ask is None:
            return

        # Passive quote, but avoid quoting nonsense negative prices.
        bid = int(math.floor(fair - mm_width))
        ask = int(math.ceil(fair + mm_width))
        bid = max(0, bid)
        ask = max(1, ask)

        bid = max(bid, best_bid + 1)
        bid = min(bid, best_ask - 1)

        ask = min(ask, best_ask - 1)
        ask = max(ask, best_bid + 1)

        if bid < best_ask and bid >= 0:
            qty = min(mm_size, self.buy_capacity(product, state, orders))
            if qty > 0:
                self.add_order(orders, product, bid, qty)

        if ask > best_bid and ask >= 1:
            qty = min(mm_size, self.sell_capacity(product, state, orders))
            if qty > 0:
                self.add_order(orders, product, ask, -qty)

    def collect_smile_points(self, state: TradingState, spot: float) -> List[Tuple[str, float, float]]:
        points: List[Tuple[str, float, float]] = []
        sqrt_t = math.sqrt(self.TTE_YEARS)

        for product, strike in self.STRIKES.items():
            if product not in state.order_depths:
                continue
            depth = state.order_depths[product]
            mid = self.mid_price(depth)
            if mid is None:
                continue

            intrinsic = max(0.0, spot - strike)

            # Skip deep intrinsic-only points and dead 0/1 far-wing points in the fit.
            if product in ("VEV_4000", "VEV_4500", "VEV_6000", "VEV_6500"):
                continue
            if mid <= intrinsic + 0.50 or mid <= 1.0:
                continue

            iv = self.implied_vol_call(mid, spot, strike, self.TTE_YEARS)
            if iv is None:
                continue

            m = math.log(strike / spot) / sqrt_t
            if 0.05 <= iv <= 1.25 and -1.0 <= m <= 1.0:
                points.append((product, m, iv))

        return points

    def option_fair_value(
        self,
        product: str,
        strike: int,
        spot: float,
        smile_points: List[Tuple[str, float, float]],
    ) -> float:
        sqrt_t = math.sqrt(self.TTE_YEARS)
        m = math.log(strike / spot) / sqrt_t

        iv = None
        # Leave-one-out fit when possible so one bad voucher does not define its own fair.
        filtered = [(x, y) for (p, x, y) in smile_points if p != product]
        if len(filtered) >= 4:
            coeffs = self.quadratic_fit(filtered)
            if coeffs is not None:
                a, b, c = coeffs
                iv = a * m * m + b * m + c

        if iv is None:
            iv = self.static_smile_iv(m)

        # Blend fitted IV with historical fallback; this avoids unstable fits in sparse/noisy books.
        iv = 0.75 * max(0.05, min(1.00, iv)) + 0.25 * self.static_smile_iv(m)
        iv = max(0.05, min(1.00, iv))
        return self.black_scholes_call(spot, float(strike), self.TTE_YEARS, iv)

    def static_smile_iv(self, m: float) -> float:
        # Round 4 Data Capsule central VEV IV is closer to 25-26%, not Round 3's 22-24%.
        # Far OTM calls at 6000/6500 need much higher fallback IV or the model underprices
        # the persistent 0/1 premium region.
        if m > 0.75:
            iv = 0.255 + 0.115 * m * m
        elif m < -1.0:
            iv = 0.255 + 0.015 * m * m
        else:
            iv = 0.255 + 0.005 * m + 0.025 * m * m
        return max(0.08, min(0.95, iv))

    def quadratic_fit(self, points: List[Tuple[float, float]]):
        # Least-squares fit y = a*x^2 + b*x + c using normal equations.
        n = len(points)
        sx = sx2 = sx3 = sx4 = 0.0
        sy = sxy = sx2y = 0.0

        for x, y in points:
            x2 = x * x
            sx += x
            sx2 += x2
            sx3 += x2 * x
            sx4 += x2 * x2
            sy += y
            sxy += x * y
            sx2y += x2 * y

        A = [
            [sx4, sx3, sx2, sx2y],
            [sx3, sx2, sx, sxy],
            [sx2, sx, float(n), sy],
        ]
        sol = self.solve_3x3(A)
        return sol

    def solve_3x3(self, aug):
        # Gaussian elimination for 3x4 augmented matrix.
        A = [[float(v) for v in row] for row in aug]
        for col in range(3):
            pivot = col
            for r in range(col + 1, 3):
                if abs(A[r][col]) > abs(A[pivot][col]):
                    pivot = r
            if abs(A[pivot][col]) < 1e-12:
                return None
            if pivot != col:
                A[col], A[pivot] = A[pivot], A[col]

            div = A[col][col]
            for j in range(col, 4):
                A[col][j] /= div

            for r in range(3):
                if r == col:
                    continue
                factor = A[r][col]
                for j in range(col, 4):
                    A[r][j] -= factor * A[col][j]

        return A[0][3], A[1][3], A[2][3]

    def norm_cdf(self, x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def black_scholes_call(self, spot: float, strike: float, t: float, vol: float) -> float:
        if t <= 0.0 or vol <= 0.0:
            return max(0.0, spot - strike)
        if spot <= 0.0 or strike <= 0.0:
            return 0.0
        sqrt_t = math.sqrt(t)
        d1 = (math.log(spot / strike) + 0.5 * vol * vol * t) / (vol * sqrt_t)
        d2 = d1 - vol * sqrt_t
        return spot * self.norm_cdf(d1) - strike * self.norm_cdf(d2)

    def implied_vol_call(self, price: float, spot: float, strike: float, t: float):
        intrinsic = max(0.0, spot - strike)
        if price <= intrinsic + 1e-6:
            return None
        if price >= spot:
            return None

        lo = 0.0001
        hi = 2.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            val = self.black_scholes_call(spot, strike, t, mid)
            if val < price:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)