"""
Benchmark Strategies for Performance Comparison

Simple buy-and-hold strategies to serve as performance baselines
for comparing active trading strategies.
"""

from typing import Dict, Any
import pandas as pd


def buy_and_hold(
    market_data: Dict[str, pd.DataFrame],
    current_time: pd.Timestamp,
    positions: Dict[str, Any],
    context: Dict[str, Any],
    params: Dict[str, Any]
) -> Dict[str, Dict]:
    """
    Buy and Hold Strategy

    Buys a specified percentage of total equity on the first day and holds.
    This serves as a baseline benchmark for comparing active strategies.

    Logic:
    - On first iteration: Buy target_allocation% of portfolio value
    - All subsequent days: Hold the position (no rebalancing)
    - Default symbol: SPY (S&P 500 ETF)

    Parameters:
    -----------
    target_allocation : float, default=1.0
        Percentage of portfolio to invest (0.0 to 1.0)
        1.0 = 100% invested, 0.6 = 60% invested, etc.
    symbol : str, default='SPY'
        Symbol to buy and hold (only used if not part of universe)

    Returns:
    --------
    Dict[str, Dict]
        Dictionary of orders keyed by symbol

    Example:
    --------
    >>> # 100% invested in SPY
    >>> params = {'target_allocation': 1.0}
    >>> result = backtester.run(
    ...     strategy=buy_and_hold,
    ...     universe=['SPY'],
    ...     strategy_params=params
    ... )

    >>> # 60% invested in QQQ (60/40 portfolio when combined with bonds)
    >>> params = {'target_allocation': 0.6}
    >>> result = backtester.run(
    ...     strategy=buy_and_hold,
    ...     universe=['QQQ'],
    ...     strategy_params=params
    ... )
    """
    # Strategy parameters
    target_allocation = params.get('target_allocation', 1.0)
    default_symbol = params.get('symbol', 'SPY')

    # Validate allocation
    if target_allocation < 0 or target_allocation > 1:
        raise ValueError(f"target_allocation must be between 0 and 1, got {target_allocation}")

    orders = {}

    # Check if this is the first iteration
    if 'initialized' not in context:
        context['initialized'] = True

        # Determine which symbols to buy
        # If market_data contains symbols, use those; otherwise use default_symbol
        symbols_to_buy = list(market_data.keys()) if market_data else [default_symbol]

        # Equal weight across all symbols
        n_symbols = len(symbols_to_buy)
        weight_per_symbol = target_allocation / n_symbols

        # Create buy orders for all symbols
        for symbol in symbols_to_buy:
            if weight_per_symbol > 0:
                orders[symbol] = {
                    'action': 'target_weight',
                    'weight': weight_per_symbol,
                    'meta': {
                        'reason': f'Buy and hold: Initial purchase at {target_allocation:.1%} allocation',
                        'strategy': 'buy_and_hold'
                    }
                }

        # Store initial purchase info
        context['purchase_date'] = current_time
        context['symbols_held'] = symbols_to_buy
        context['allocation'] = target_allocation

    # After first day: Hold (return empty orders dict - no trading)
    # The positions will naturally drift with market movements

    return orders


def buy_and_hold_rebalanced(
    market_data: Dict[str, pd.DataFrame],
    current_time: pd.Timestamp,
    positions: Dict[str, Any],
    context: Dict[str, Any],
    params: Dict[str, Any]
) -> Dict[str, Dict]:
    """
    Buy and Hold with Periodic Rebalancing

    Maintains a constant allocation by rebalancing at specified intervals.
    Useful for benchmarking strategies that might benefit from rebalancing drift.

    Logic:
    - Maintains target_allocation% of portfolio in specified symbols
    - Rebalances at specified frequency (daily, weekly, monthly, quarterly)
    - Equal weight across all symbols in universe

    Parameters:
    -----------
    target_allocation : float, default=1.0
        Percentage of portfolio to invest (0.0 to 1.0)
    rebalance_frequency : str, default='monthly'
        How often to rebalance: 'daily', 'weekly', 'monthly', 'quarterly', 'yearly'
    symbol : str, default='SPY'
        Default symbol if not using universe

    Returns:
    --------
    Dict[str, Dict]
        Dictionary of orders keyed by symbol

    Example:
    --------
    >>> # Monthly rebalanced portfolio
    >>> params = {'target_allocation': 1.0, 'rebalance_frequency': 'monthly'}
    >>> result = backtester.run(
    ...     strategy=buy_and_hold_rebalanced,
    ...     universe=['SPY', 'AGG'],  # 60/40 stocks/bonds
    ...     strategy_params=params
    ... )
    """
    # Strategy parameters
    target_allocation = params.get('target_allocation', 1.0)
    rebalance_freq = params.get('rebalance_frequency', 'monthly').lower()
    default_symbol = params.get('symbol', 'SPY')

    # Validate allocation
    if target_allocation < 0 or target_allocation > 1:
        raise ValueError(f"target_allocation must be between 0 and 1, got {target_allocation}")

    orders = {}

    # Initialize on first call
    if 'last_rebalance' not in context:
        context['last_rebalance'] = None

    # Determine if we need to rebalance
    should_rebalance = False

    if context['last_rebalance'] is None:
        # First iteration - always rebalance
        should_rebalance = True
    else:
        # Check if rebalance period has elapsed
        last_rebal = context['last_rebalance']

        if rebalance_freq == 'daily':
            should_rebalance = True
        elif rebalance_freq == 'weekly':
            should_rebalance = (current_time - last_rebal).days >= 7
        elif rebalance_freq == 'monthly':
            should_rebalance = current_time.month != last_rebal.month or current_time.year != last_rebal.year
        elif rebalance_freq == 'quarterly':
            should_rebalance = (current_time.month - 1) // 3 != (last_rebal.month - 1) // 3 or current_time.year != last_rebal.year
        elif rebalance_freq == 'yearly':
            should_rebalance = current_time.year != last_rebal.year
        else:
            raise ValueError(f"Invalid rebalance_frequency: {rebalance_freq}")

    # Rebalance if needed
    if should_rebalance:
        # Determine which symbols to hold
        symbols_to_hold = list(market_data.keys()) if market_data else [default_symbol]

        # Equal weight across all symbols
        n_symbols = len(symbols_to_hold)
        weight_per_symbol = target_allocation / n_symbols

        # Create rebalancing orders for all symbols
        for symbol in symbols_to_hold:
            if weight_per_symbol > 0:
                orders[symbol] = {
                    'action': 'target_weight',
                    'weight': weight_per_symbol,
                    'meta': {
                        'reason': f'Rebalance to {target_allocation:.1%} allocation',
                        'strategy': 'buy_and_hold_rebalanced',
                        'rebalance_frequency': rebalance_freq
                    }
                }

        # Update last rebalance date
        context['last_rebalance'] = current_time

    return orders


def sixty_forty_portfolio(
    market_data: Dict[str, pd.DataFrame],
    current_time: pd.Timestamp,
    positions: Dict[str, Any],
    context: Dict[str, Any],
    params: Dict[str, Any]
) -> Dict[str, Dict]:
    """
    Classic 60/40 Stock/Bond Portfolio

    Traditional balanced portfolio: 60% stocks (SPY), 40% bonds (AGG).
    Rebalances at specified intervals to maintain target allocation.

    Logic:
    - 60% allocation to stocks (default: SPY)
    - 40% allocation to bonds (default: AGG)
    - Rebalances at specified frequency

    Parameters:
    -----------
    stock_allocation : float, default=0.6
        Percentage allocated to stocks (bonds = 1 - stock_allocation)
    stock_symbol : str, default='SPY'
        Stock ETF to use
    bond_symbol : str, default='AGG'
        Bond ETF to use
    rebalance_frequency : str, default='quarterly'
        How often to rebalance: 'monthly', 'quarterly', 'yearly'

    Returns:
    --------
    Dict[str, Dict]
        Dictionary of orders keyed by symbol

    Example:
    --------
    >>> # Classic 60/40 portfolio with quarterly rebalancing
    >>> result = backtester.run(
    ...     strategy=sixty_forty_portfolio,
    ...     universe=['SPY', 'AGG'],
    ...     strategy_params={'stock_allocation': 0.6, 'rebalance_frequency': 'quarterly'}
    ... )
    """
    # Strategy parameters
    stock_allocation = params.get('stock_allocation', 0.6)
    stock_symbol = params.get('stock_symbol', 'SPY')
    bond_symbol = params.get('bond_symbol', 'AGG')
    rebalance_freq = params.get('rebalance_frequency', 'quarterly').lower()

    # Validate allocation
    if stock_allocation < 0 or stock_allocation > 1:
        raise ValueError(f"stock_allocation must be between 0 and 1, got {stock_allocation}")

    bond_allocation = 1.0 - stock_allocation

    orders = {}

    # Initialize on first call
    if 'last_rebalance' not in context:
        context['last_rebalance'] = None

    # Determine if we need to rebalance
    should_rebalance = False

    if context['last_rebalance'] is None:
        should_rebalance = True
    else:
        last_rebal = context['last_rebalance']

        if rebalance_freq == 'monthly':
            should_rebalance = current_time.month != last_rebal.month or current_time.year != last_rebal.year
        elif rebalance_freq == 'quarterly':
            should_rebalance = (current_time.month - 1) // 3 != (last_rebal.month - 1) // 3 or current_time.year != last_rebal.year
        elif rebalance_freq == 'yearly':
            should_rebalance = current_time.year != last_rebal.year
        else:
            raise ValueError(f"Invalid rebalance_frequency: {rebalance_freq}")

    # Rebalance if needed
    if should_rebalance:
        # Check which symbols are available in market_data
        available_symbols = list(market_data.keys()) if market_data else []

        # Determine actual symbols to use
        actual_stock = stock_symbol if stock_symbol in available_symbols or not available_symbols else available_symbols[0]
        actual_bond = bond_symbol if bond_symbol in available_symbols or len(available_symbols) < 2 else available_symbols[1] if len(available_symbols) > 1 else bond_symbol

        # Create orders for stock and bond
        if stock_allocation > 0:
            orders[actual_stock] = {
                'action': 'target_weight',
                'weight': stock_allocation,
                'meta': {
                    'reason': f'{stock_allocation:.0%} stock allocation',
                    'strategy': '60_40_portfolio',
                    'asset_class': 'stocks'
                }
            }

        if bond_allocation > 0:
            orders[actual_bond] = {
                'action': 'target_weight',
                'weight': bond_allocation,
                'meta': {
                    'reason': f'{bond_allocation:.0%} bond allocation',
                    'strategy': '60_40_portfolio',
                    'asset_class': 'bonds'
                }
            }

        # Update last rebalance date
        context['last_rebalance'] = current_time
        context['stock_symbol'] = actual_stock
        context['bond_symbol'] = actual_bond

    return orders
