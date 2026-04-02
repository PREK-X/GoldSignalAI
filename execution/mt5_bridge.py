"""
GoldSignalAI -- execution/mt5_bridge.py
========================================
Stage 11: Platform-aware MT5 execution bridge.

On Linux (current dev environment):
  - Simulation mode -- logs orders, returns mock fills
  - All actions prefixed with [SIMULATION]

On Windows / VPS with MetaTrader5 Python package:
  - Real MT5 execution via mt5.order_send()
  - Full connection lifecycle, retry logic, error handling
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from config import Config

logger = logging.getLogger(__name__)


@dataclass
class OrderResult:
    """Result of an order placement or modification."""
    success: bool
    ticket: int          # MT5 ticket or mock ticket (timestamp-based)
    fill_price: float    # Actual fill price (or entry price in simulation)
    status: str          # "filled", "rejected", "error", "simulated"
    message: str = ""


@dataclass
class PositionInfo:
    """Snapshot of an open position."""
    ticket: int
    symbol: str
    direction: str       # "BUY" or "SELL"
    volume: float        # Lot size
    open_price: float
    sl: float
    tp: float
    open_time: datetime
    profit: float = 0.0
    comment: str = ""


@dataclass
class AccountInfo:
    """MT5 account summary."""
    balance: float
    equity: float
    margin: float
    free_margin: float
    margin_level: float = 0.0


class MT5Bridge:
    """
    Platform-aware execution bridge.

    Detects MetaTrader5 availability at init time:
      - "mt5"        -> real orders via MetaTrader5 Python API
      - "simulation" -> mock fills logged locally
    """

    def __init__(self):
        self.platform = self._detect_platform()
        self._connected = False
        self._mt5 = None  # MetaTrader5 module reference (real mode only)

        # Simulation state
        self._sim_positions: dict[int, PositionInfo] = {}
        self._sim_ticket_counter = 100_000

    # ── Platform detection ────────────────────────────────────────────────

    @staticmethod
    def _detect_platform() -> str:
        try:
            import MetaTrader5 as mt5  # noqa: F401
            return "mt5"
        except ImportError:
            return "simulation"

    @property
    def is_simulation(self) -> bool:
        return self.platform == "simulation"

    # ── Connection lifecycle ──────────────────────────────────────────────

    def connect(self) -> bool:
        if self.is_simulation:
            logger.info("[SIMULATION] MT5 bridge connected (simulation mode)")
            self._connected = True
            return True

        import MetaTrader5 as mt5
        self._mt5 = mt5

        for attempt in range(1, Config.MT5_RETRY_ATTEMPTS + 1):
            if not mt5.initialize():
                err = mt5.last_error()
                logger.warning(
                    "MT5 initialize failed (attempt %d/%d): %s",
                    attempt, Config.MT5_RETRY_ATTEMPTS, err,
                )
                if attempt < Config.MT5_RETRY_ATTEMPTS:
                    time.sleep(Config.MT5_RETRY_DELAY_S)
                continue

            if Config.MT5_LOGIN and Config.MT5_PASSWORD:
                authorized = mt5.login(
                    login=Config.MT5_LOGIN,
                    password=Config.MT5_PASSWORD,
                    server=Config.MT5_SERVER,
                )
                if not authorized:
                    err = mt5.last_error()
                    logger.warning(
                        "MT5 login failed (attempt %d/%d): %s",
                        attempt, Config.MT5_RETRY_ATTEMPTS, err,
                    )
                    mt5.shutdown()
                    if attempt < Config.MT5_RETRY_ATTEMPTS:
                        time.sleep(Config.MT5_RETRY_DELAY_S)
                    continue

            self._connected = True
            logger.info("MT5 bridge connected (real mode, server=%s)", Config.MT5_SERVER)
            return True

        logger.error("MT5 connection failed after %d attempts", Config.MT5_RETRY_ATTEMPTS)
        return False

    def disconnect(self) -> None:
        if self.is_simulation:
            logger.info("[SIMULATION] MT5 bridge disconnected")
            self._connected = False
            return

        if self._mt5 is not None:
            self._mt5.shutdown()
            logger.info("MT5 bridge disconnected")
        self._connected = False

    # ── Order placement ───────────────────────────────────────────────────

    def place_order(
        self,
        symbol: str,
        direction: str,
        volume: float,
        sl_price: float,
        tp_price: float,
        comment: str = "GoldSignalAI",
    ) -> OrderResult:
        if self.is_simulation:
            return self._sim_place_order(symbol, direction, volume, sl_price, tp_price, comment)
        return self._mt5_place_order(symbol, direction, volume, sl_price, tp_price, comment)

    def _sim_place_order(
        self, symbol, direction, volume, sl_price, tp_price, comment
    ) -> OrderResult:
        self._sim_ticket_counter += 1
        ticket = self._sim_ticket_counter

        # Use midpoint of SL and TP as simulated entry
        entry_price = (sl_price + tp_price) / 2

        pos = PositionInfo(
            ticket=ticket,
            symbol=symbol,
            direction=direction,
            volume=volume,
            open_price=entry_price,
            sl=sl_price,
            tp=tp_price,
            open_time=datetime.now(timezone.utc),
            comment=comment,
        )
        self._sim_positions[ticket] = pos

        logger.info(
            "[SIMULATION] Order placed: %s %s %.2f lots @ %.2f | SL=%.2f TP=%.2f | ticket=%d",
            direction, symbol, volume, entry_price, sl_price, tp_price, ticket,
        )
        return OrderResult(
            success=True,
            ticket=ticket,
            fill_price=entry_price,
            status="simulated",
            message=f"Simulated {direction} {volume} {symbol}",
        )

    def _mt5_place_order(
        self, symbol, direction, volume, sl_price, tp_price, comment
    ) -> OrderResult:
        mt5 = self._mt5
        order_type = mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return OrderResult(
                success=False, ticket=0, fill_price=0.0,
                status="error", message=f"No tick data for {symbol}",
            )

        price = tick.ask if direction == "BUY" else tick.bid

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": sl_price,
            "tp": tp_price,
            "deviation": Config.MT5_MAX_SLIPPAGE,
            "magic": Config.MT5_MAGIC_NUMBER,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        for attempt in range(1, Config.MT5_RETRY_ATTEMPTS + 1):
            result = mt5.order_send(request)
            if result is None:
                err = mt5.last_error()
                logger.warning("order_send returned None (attempt %d): %s", attempt, err)
                if attempt < Config.MT5_RETRY_ATTEMPTS:
                    time.sleep(Config.MT5_RETRY_DELAY_S)
                continue

            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(
                    "MT5 order filled: %s %s %.2f lots @ %.5f | ticket=%d",
                    direction, symbol, volume, result.price, result.order,
                )
                return OrderResult(
                    success=True,
                    ticket=result.order,
                    fill_price=result.price,
                    status="filled",
                    message=f"Filled at {result.price}",
                )

            logger.warning(
                "order_send retcode=%d (attempt %d): %s",
                result.retcode, attempt, result.comment,
            )
            if attempt < Config.MT5_RETRY_ATTEMPTS:
                time.sleep(Config.MT5_RETRY_DELAY_S)

        return OrderResult(
            success=False, ticket=0, fill_price=0.0,
            status="rejected",
            message=f"Order rejected after {Config.MT5_RETRY_ATTEMPTS} attempts",
        )

    # ── Order closing ─────────────────────────────────────────────────────

    def close_order(self, ticket: int) -> OrderResult:
        if self.is_simulation:
            return self._sim_close_order(ticket)
        return self._mt5_close_order(ticket)

    def _sim_close_order(self, ticket: int) -> OrderResult:
        pos = self._sim_positions.pop(ticket, None)
        if pos is None:
            return OrderResult(
                success=False, ticket=ticket, fill_price=0.0,
                status="error", message=f"No simulated position with ticket {ticket}",
            )
        logger.info(
            "[SIMULATION] Order closed: ticket=%d %s %s @ %.2f",
            ticket, pos.direction, pos.symbol, pos.open_price,
        )
        return OrderResult(
            success=True, ticket=ticket, fill_price=pos.open_price,
            status="simulated", message="Simulated close",
        )

    def _mt5_close_order(self, ticket: int) -> OrderResult:
        mt5 = self._mt5
        position = mt5.positions_get(ticket=ticket)
        if not position:
            return OrderResult(
                success=False, ticket=ticket, fill_price=0.0,
                status="error", message=f"Position {ticket} not found",
            )

        pos = position[0]
        close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        symbol = pos.symbol
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return OrderResult(
                success=False, ticket=ticket, fill_price=0.0,
                status="error", message=f"No tick data for {symbol}",
            )

        price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": pos.volume,
            "type": close_type,
            "position": ticket,
            "price": price,
            "deviation": Config.MT5_MAX_SLIPPAGE,
            "magic": Config.MT5_MAGIC_NUMBER,
            "comment": "GoldSignalAI close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info("MT5 position %d closed @ %.5f", ticket, result.price)
            return OrderResult(
                success=True, ticket=ticket, fill_price=result.price,
                status="filled", message="Position closed",
            )

        err = mt5.last_error() if result is None else result.comment
        return OrderResult(
            success=False, ticket=ticket, fill_price=0.0,
            status="rejected", message=f"Close failed: {err}",
        )

    # ── Position queries ──────────────────────────────────────────────────

    def get_position(self, ticket: int) -> Optional[PositionInfo]:
        if self.is_simulation:
            return self._sim_positions.get(ticket)

        mt5 = self._mt5
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return None

        p = positions[0]
        return PositionInfo(
            ticket=p.ticket,
            symbol=p.symbol,
            direction="BUY" if p.type == mt5.ORDER_TYPE_BUY else "SELL",
            volume=p.volume,
            open_price=p.price_open,
            sl=p.sl,
            tp=p.tp,
            open_time=datetime.fromtimestamp(p.time, tz=timezone.utc),
            profit=p.profit,
            comment=p.comment,
        )

    def get_open_positions(self) -> list[PositionInfo]:
        if self.is_simulation:
            return list(self._sim_positions.values())

        mt5 = self._mt5
        positions = mt5.positions_get(symbol=Config.MT5_SYMBOL)
        if not positions:
            return []

        return [
            PositionInfo(
                ticket=p.ticket,
                symbol=p.symbol,
                direction="BUY" if p.type == mt5.ORDER_TYPE_BUY else "SELL",
                volume=p.volume,
                open_price=p.price_open,
                sl=p.sl,
                tp=p.tp,
                open_time=datetime.fromtimestamp(p.time, tz=timezone.utc),
                profit=p.profit,
                comment=p.comment,
            )
            for p in positions
            if p.magic == Config.MT5_MAGIC_NUMBER
        ]

    def get_account_info(self) -> Optional[AccountInfo]:
        if self.is_simulation:
            return AccountInfo(
                balance=Config.CHALLENGE_ACCOUNT_SIZE,
                equity=Config.CHALLENGE_ACCOUNT_SIZE,
                margin=0.0,
                free_margin=Config.CHALLENGE_ACCOUNT_SIZE,
            )

        mt5 = self._mt5
        info = mt5.account_info()
        if info is None:
            return None

        return AccountInfo(
            balance=info.balance,
            equity=info.equity,
            margin=info.margin,
            free_margin=info.margin_free,
            margin_level=info.margin_level,
        )

    # ── SL modification ───────────────────────────────────────────────────

    def modify_sl(self, ticket: int, new_sl: float) -> bool:
        if self.is_simulation:
            pos = self._sim_positions.get(ticket)
            if pos is None:
                return False
            logger.info(
                "[SIMULATION] SL modified: ticket=%d  %.2f -> %.2f",
                ticket, pos.sl, new_sl,
            )
            pos.sl = new_sl
            return True

        mt5 = self._mt5
        position = mt5.positions_get(ticket=ticket)
        if not position:
            return False

        p = position[0]
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": p.symbol,
            "position": ticket,
            "sl": new_sl,
            "tp": p.tp,
            "magic": Config.MT5_MAGIC_NUMBER,
        }

        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info("MT5 SL modified: ticket=%d -> %.5f", ticket, new_sl)
            return True

        err = mt5.last_error() if result is None else result.comment
        logger.warning("SL modify failed for ticket %d: %s", ticket, err)
        return False
