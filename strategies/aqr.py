"""
AQR Capital Management Style Quantitative Strategies

Implementation of factor-based strategies disclosed in AQR research papers:
- Quality Minus Junk (QMJ) - Asness, Frazzini, Pedersen (2014)
- Value Everywhere - Multi-factor value approach
- Betting Against Beta (BAB) - Frazzini, Pedersen (2014)
- Quality + Value + Momentum - Combined multi-factor approach

All strategies based on publicly available AQR research and datasets.

References:
- Quality Minus Junk: https://www.aqr.com/Insights/Research/Working-Paper/Quality-Minus-Junk
- Value and Momentum Everywhere: https://www.aqr.com/Insights/Datasets/Value-and-Momentum-Everywhere-Factors-Monthly
- Betting Against Beta: https://www.aqr.com/Insights/Research/Journal-Article/Betting-Against-Beta
"""

from typing import Dict, Any
import pandas as pd
import numpy as np
from backt.signal import TechnicalIndicators
from backt.data.fundamentals import FundamentalsLoader, calculate_quality_score, calculate_value_score


def quality_minus_junk(
    market_data: Dict[str, pd.DataFrame],
    current_time: pd.Timestamp,
    positions: Dict[str, Any],
    context: Dict[str, Any],
    params: Dict[str, Any]
) -> Dict[str, Dict]:
    """
    Quality Minus Junk (QMJ) Strategy - AQR Capital Management

    Based on: "Quality Minus Junk" (Asness, Frazzini, Pedersen, 2014)

    Logic:
    - Calculate composite quality score for each stock
    - Quality = Profitability + Growth + Safety + Payout
    - LONG top 30% quality stocks (high quality)
    - SHORT bottom 30% quality stocks (junk)
    - Equal or value-weighted within portfolios

    Quality Components:
    - Profitability: ROE, ROA, Margins
    - Growth: Revenue growth, Earnings growth
    - Safety: Low leverage, High liquidity ratios
    - Payout: Positive free cash flow

    Parameters:
    -----------
    top_percentile : float, default=0.30
        Top percentile for quality stocks to long
    bottom_percentile : float, default=0.30
        Bottom percentile for junk stocks to short
    allow_short : bool, default=True
        If True, short junk stocks; if False, long-only
    rebalance_frequency : str, default='monthly'
        How often to rebalance ('monthly', 'quarterly')
    weighting : str, default='equal'
        Portfolio weighting: 'equal' or 'value'

    Returns:
    --------
    Dict[str, Dict]
        Dictionary of orders keyed by symbol

    Example:
    --------
    >>> params = {'top_percentile': 0.30, 'allow_short': True}
    >>> result = backtester.run(
    ...     strategy=quality_minus_junk,
    ...     universe=['AAPL', 'MSFT', 'GOOGL', ... ],  # Large universe recommended
    ...     strategy_params=params
    ... )
    """
    # Strategy parameters
    top_pct = params.get('top_percentile', 0.30)
    bottom_pct = params.get('bottom_percentile', 0.30)
    allow_short = params.get('allow_short', True)
    weighting = params.get('weighting', 'equal')

    # Initialize fundamentals loader in context if needed
    if 'fundamentals_loader' not in context:
        context['fundamentals_loader'] = FundamentalsLoader()

    loader = context['fundamentals_loader']

    # Calculate quality scores for all symbols
    quality_scores = {}
    market_caps = {}

    for symbol in market_data.keys():
        fundamentals = loader.get_fundamentals(symbol)
        if fundamentals is None:
            continue

        quality = calculate_quality_score(fundamentals)
        if quality is not None:
            quality_scores[symbol] = quality
            market_caps[symbol] = fundamentals.get('market_cap', 1)

    # Need at least some stocks with quality scores
    if len(quality_scores) < 5:
        context['error'] = 'Insufficient fundamental data available'
        return {}

    # Rank by quality
    sorted_by_quality = sorted(quality_scores.items(), key=lambda x: x[1], reverse=True)

    # Select top and bottom percentiles
    n_stocks = len(sorted_by_quality)
    n_top = max(1, int(n_stocks * top_pct))
    n_bottom = max(1, int(n_stocks * bottom_pct))

    high_quality = [symbol for symbol, _ in sorted_by_quality[:n_top]]
    junk = [symbol for symbol, _ in sorted_by_quality[-n_bottom:]] if allow_short else []

    orders = {}
    signals = {}

    # Calculate position sizing
    total_positions = len(high_quality) + len(junk)

    if total_positions > 0:
        if weighting == 'equal':
            # Equal weight - pure 1/N allocation
            weight_per_position = 1.0 / total_positions

            # Long high quality
            for symbol in high_quality:
                orders[symbol] = {
                    'action': 'target_weight',
                    'weight': weight_per_position
                }
                signals[symbol] = f'HIGH_QUALITY (score: {quality_scores[symbol]:.3f})'

            # Short junk
            for symbol in junk:
                orders[symbol] = {
                    'action': 'target_weight',
                    'weight': -weight_per_position
                }
                signals[symbol] = f'JUNK (score: {quality_scores[symbol]:.3f})'

        else:  # value-weighted
            # Calculate total market cap for each side
            total_long_mcap = sum(market_caps.get(s, 1) for s in high_quality)
            total_short_mcap = sum(market_caps.get(s, 1) for s in junk) if junk else 1

            # Long high quality (value-weighted)
            for symbol in high_quality:
                mcap_weight = market_caps.get(symbol, 1) / total_long_mcap
                weight = mcap_weight * 0.5  # 50% allocated to longs
                orders[symbol] = {
                    'action': 'target_weight',
                    'weight': weight
                }
                signals[symbol] = f'HIGH_QUALITY (score: {quality_scores[symbol]:.3f})'

            # Short junk (value-weighted)
            for symbol in junk:
                mcap_weight = market_caps.get(symbol, 1) / total_short_mcap
                weight = mcap_weight * 0.5  # 50% allocated to shorts
                orders[symbol] = {
                    'action': 'target_weight',
                    'weight': -weight
                }
                signals[symbol] = f'JUNK (score: {quality_scores[symbol]:.3f})'

    # Close positions not in selected portfolios
    active_positions = high_quality + junk
    for symbol in market_data.keys():
        if symbol not in active_positions:
            if symbol in positions and hasattr(positions[symbol], 'qty'):
                if positions[symbol].qty != 0:
                    orders[symbol] = {'action': 'close'}

    # Store strategy state
    context['signals'] = signals
    context['quality_scores'] = quality_scores
    context['high_quality'] = high_quality
    context['junk'] = junk

    return orders


def quality_long_only(
    market_data: Dict[str, pd.DataFrame],
    current_time: pd.Timestamp,
    positions: Dict[str, Any],
    context: Dict[str, Any],
    params: Dict[str, Any]
) -> Dict[str, Dict]:
    """
    Quality Long-Only Strategy

    Similar to QMJ but long-only (no shorting).
    Goes long the highest quality stocks.

    Parameters:
    -----------
    top_percentile : float, default=0.30
        Top percentile for quality stocks to long

    Returns:
    --------
    Dict[str, Dict]
        Dictionary of orders keyed by symbol
    """
    # Force long-only
    params['allow_short'] = False
    return quality_minus_junk(market_data, current_time, positions, context, params)


def value_everywhere(
    market_data: Dict[str, pd.DataFrame],
    current_time: pd.Timestamp,
    positions: Dict[str, Any],
    context: Dict[str, Any],
    params: Dict[str, Any]
) -> Dict[str, Dict]:
    """
    Value Everywhere Strategy - AQR Capital Management

    Based on: "Value and Momentum Everywhere" (Asness, Moskowitz, Pedersen, 2013)

    Logic:
    - Calculate composite value score for each stock
    - Value = Average of Book/Price, Earnings/Price, CF/Price
    - LONG top 30% value stocks (cheap)
    - SHORT bottom 30% value stocks (expensive) if allowed
    - Equal or value-weighted within portfolios

    Parameters:
    -----------
    top_percentile : float, default=0.30
        Top percentile for value stocks to long
    bottom_percentile : float, default=0.30
        Bottom percentile for growth stocks to short
    allow_short : bool, default=True
        If True, short expensive stocks; if False, long-only

    Returns:
    --------
    Dict[str, Dict]
        Dictionary of orders keyed by symbol
    """
    # Strategy parameters
    top_pct = params.get('top_percentile', 0.30)
    bottom_pct = params.get('bottom_percentile', 0.30)
    allow_short = params.get('allow_short', True)

    # Initialize fundamentals loader
    if 'fundamentals_loader' not in context:
        context['fundamentals_loader'] = FundamentalsLoader()

    loader = context['fundamentals_loader']

    # Calculate value scores
    value_scores = {}

    for symbol in market_data.keys():
        fundamentals = loader.get_fundamentals(symbol)
        if fundamentals is None:
            continue

        value = calculate_value_score(fundamentals)
        if value is not None:
            value_scores[symbol] = value

    if len(value_scores) < 5:
        context['error'] = 'Insufficient fundamental data'
        return {}

    # Rank by value (higher = cheaper/better value)
    sorted_by_value = sorted(value_scores.items(), key=lambda x: x[1], reverse=True)

    n_stocks = len(sorted_by_value)
    n_top = max(1, int(n_stocks * top_pct))
    n_bottom = max(1, int(n_stocks * bottom_pct))

    value_stocks = [symbol for symbol, _ in sorted_by_value[:n_top]]
    growth_stocks = [symbol for symbol, _ in sorted_by_value[-n_bottom:]] if allow_short else []

    orders = {}
    signals = {}

    # Position sizing - pure 1/N allocation
    total_positions = len(value_stocks) + len(growth_stocks)

    if total_positions > 0:
        weight_per_position = 1.0 / total_positions

        # Long value
        for symbol in value_stocks:
            orders[symbol] = {
                'action': 'target_weight',
                'weight': weight_per_position
            }
            signals[symbol] = f'VALUE (score: {value_scores[symbol]:.4f})'

        # Short growth/expensive
        for symbol in growth_stocks:
            orders[symbol] = {
                'action': 'target_weight',
                'weight': -weight_per_position
            }
            signals[symbol] = f'EXPENSIVE (score: {value_scores[symbol]:.4f})'

    # Close other positions
    active = value_stocks + growth_stocks
    for symbol in market_data.keys():
        if symbol not in active:
            if symbol in positions and hasattr(positions[symbol], 'qty'):
                if positions[symbol].qty != 0:
                    orders[symbol] = {'action': 'close'}

    context['signals'] = signals
    context['value_scores'] = value_scores
    context['value_stocks'] = value_stocks
    context['growth_stocks'] = growth_stocks

    return orders


def betting_against_beta(
    market_data: Dict[str, pd.DataFrame],
    current_time: pd.Timestamp,
    positions: Dict[str, Any],
    context: Dict[str, Any],
    params: Dict[str, Any]
) -> Dict[str, Dict]:
    """
    Betting Against Beta (BAB) Strategy - AQR Capital Management

    Based on: "Betting Against Beta" (Frazzini, Pedersen, 2014)

    Logic:
    - Calculate beta for each stock (vs market)
    - LONG low-beta stocks (stable, defensive)
    - SHORT high-beta stocks (volatile, aggressive)
    - Use leverage to make portfolios market-neutral

    Key Insight: Low-beta stocks historically outperform high-beta stocks
    on a risk-adjusted basis.

    Parameters:
    -----------
    beta_period : int, default=60
        Months for beta calculation (60 months = 5 years)
    top_percentile : float, default=0.30
        Top percentile for low-beta stocks
    bottom_percentile : float, default=0.30
        Bottom percentile for high-beta stocks
    allow_short : bool, default=True
        If True, short high-beta; if False, long-only low-beta

    Returns:
    --------
    Dict[str, Dict]
        Dictionary of orders keyed by symbol
    """
    # Strategy parameters
    beta_period = params.get('beta_period', 252)  # 252 trading days ~ 1 year
    top_pct = params.get('top_percentile', 0.30)
    bottom_pct = params.get('bottom_percentile', 0.30)
    allow_short = params.get('allow_short', True)

    # Initialize fundamentals loader
    if 'fundamentals_loader' not in context:
        context['fundamentals_loader'] = FundamentalsLoader()

    loader = context['fundamentals_loader']

    # Calculate beta for each stock
    betas = {}

    for symbol, data in market_data.items():
        # Try to get beta from fundamentals first (quicker)
        fundamentals = loader.get_fundamentals(symbol)
        if fundamentals and fundamentals.get('beta') is not None:
            betas[symbol] = fundamentals['beta']
        else:
            # Calculate beta from returns if fundamentals not available
            if len(data) >= beta_period:
                try:
                    returns = data['close'].pct_change().dropna()
                    if len(returns) >= beta_period:
                        # Simple beta calculation (would need market returns for proper calc)
                        # Using volatility as proxy - lower vol = lower beta
                        volatility = returns.rolling(beta_period).std().iloc[-1]
                        # Normalize to beta scale (0.5 to 1.5 typical range)
                        beta_proxy = volatility / returns.std()
                        betas[symbol] = beta_proxy
                except:
                    continue

    if len(betas) < 5:
        context['error'] = 'Insufficient beta data'
        return {}

    # Rank by beta
    sorted_by_beta = sorted(betas.items(), key=lambda x: x[1])

    n_stocks = len(sorted_by_beta)
    n_low_beta = max(1, int(n_stocks * top_pct))
    n_high_beta = max(1, int(n_stocks * bottom_pct))

    low_beta_stocks = [symbol for symbol, _ in sorted_by_beta[:n_low_beta]]
    high_beta_stocks = [symbol for symbol, _ in sorted_by_beta[-n_high_beta:]] if allow_short else []

    orders = {}
    signals = {}

    # Position sizing
    total_positions = len(low_beta_stocks) + len(high_beta_stocks)

    if total_positions > 0:
        weight_per_position = 1.0 / total_positions  # Pure 1/N allocation

        # Long low-beta
        for symbol in low_beta_stocks:
            orders[symbol] = {
                'action': 'target_weight',
                'weight': weight_per_position
            }
            signals[symbol] = f'LOW_BETA (β={betas[symbol]:.3f})'

        # Short high-beta
        for symbol in high_beta_stocks:
            orders[symbol] = {
                'action': 'target_weight',
                'weight': -weight_per_position
            }
            signals[symbol] = f'HIGH_BETA (β={betas[symbol]:.3f})'

    # Close other positions
    active = low_beta_stocks + high_beta_stocks
    for symbol in market_data.keys():
        if symbol not in active:
            if symbol in positions and hasattr(positions[symbol], 'qty'):
                if positions[symbol].qty != 0:
                    orders[symbol] = {'action': 'close'}

    context['signals'] = signals
    context['betas'] = betas
    context['low_beta'] = low_beta_stocks
    context['high_beta'] = high_beta_stocks

    return orders


def defensive_equity(
    market_data: Dict[str, pd.DataFrame],
    current_time: pd.Timestamp,
    positions: Dict[str, Any],
    context: Dict[str, Any],
    params: Dict[str, Any]
) -> Dict[str, Dict]:
    """
    Defensive Equity Strategy (Long-Only BAB)

    Long-only version of Betting Against Beta.
    Focuses on low-beta, low-volatility, defensive stocks.

    Parameters:
    -----------
    top_percentile : float, default=0.50
        Top percentile for defensive stocks to long

    Returns:
    --------
    Dict[str, Dict]
        Dictionary of orders keyed by symbol
    """
    params['allow_short'] = False
    params['top_percentile'] = params.get('top_percentile', 0.50)
    return betting_against_beta(market_data, current_time, positions, context, params)


def quality_value_momentum(
    market_data: Dict[str, pd.DataFrame],
    current_time: pd.Timestamp,
    positions: Dict[str, Any],
    context: Dict[str, Any],
    params: Dict[str, Any]
) -> Dict[str, Dict]:
    """
    Quality + Value + Momentum Multi-Factor Strategy - AQR's "New Core"

    Combines three factors as described in AQR's research:
    - Quality: High profitability, low leverage
    - Value: Cheap on fundamentals
    - Momentum: 12-month price momentum

    Logic:
    - Calculate composite score = 40% Quality + 30% Value + 30% Momentum
    - Long top 30% on combined score
    - Short bottom 30% if allowed

    Parameters:
    -----------
    quality_weight : float, default=0.40
        Weight for quality factor
    value_weight : float, default=0.30
        Weight for value factor
    momentum_weight : float, default=0.30
        Weight for momentum factor
    top_percentile : float, default=0.30
        Top percentile to long
    bottom_percentile : float, default=0.30
        Bottom percentile to short
    allow_short : bool, default=True
        Allow short positions
    momentum_period : int, default=252
        Days for momentum calculation (252 ~ 12 months)
    skip_recent_days : int, default=20
        Days to skip at end for momentum (avoid reversal)

    Returns:
    --------
    Dict[str, Dict]
        Dictionary of orders keyed by symbol
    """
    # Strategy parameters
    q_weight = params.get('quality_weight', 0.40)
    v_weight = params.get('value_weight', 0.30)
    m_weight = params.get('momentum_weight', 0.30)
    top_pct = params.get('top_percentile', 0.30)
    bottom_pct = params.get('bottom_percentile', 0.30)
    allow_short = params.get('allow_short', True)
    momentum_period = params.get('momentum_period', 252)
    skip_recent = params.get('skip_recent_days', 20)

    # Initialize fundamentals loader
    if 'fundamentals_loader' not in context:
        context['fundamentals_loader'] = FundamentalsLoader()

    loader = context['fundamentals_loader']

    # Calculate all three factor scores
    combined_scores = {}

    for symbol, data in market_data.items():
        scores = {}

        # Quality score
        fundamentals = loader.get_fundamentals(symbol)
        if fundamentals:
            quality = calculate_quality_score(fundamentals)
            if quality is not None:
                scores['quality'] = quality

            # Value score
            value = calculate_value_score(fundamentals)
            if value is not None:
                scores['value'] = value

        # Momentum score
        if len(data) >= momentum_period + skip_recent:
            try:
                # 12-month return, skipping last month
                end_idx = -skip_recent if skip_recent > 0 else None
                start_price = data['close'].iloc[-(momentum_period + skip_recent)]
                end_price = data['close'].iloc[end_idx] if end_idx else data['close'].iloc[-1]
                momentum_return = (end_price - start_price) / start_price
                scores['momentum'] = momentum_return
            except:
                pass

        # Need at least 2 of 3 factors
        if len(scores) >= 2:
            # Normalize each score to 0-1 range using rank percentile
            # (This would ideally be done across all stocks, simplified here)
            combined = 0
            weight_sum = 0

            if 'quality' in scores:
                combined += q_weight * scores['quality']
                weight_sum += q_weight
            if 'value' in scores:
                combined += v_weight * scores['value']
                weight_sum += v_weight
            if 'momentum' in scores:
                # Normalize momentum to 0-1 range (assume -50% to +100% range)
                norm_momentum = (scores['momentum'] + 0.5) / 1.5
                combined += m_weight * max(0, min(1, norm_momentum))
                weight_sum += m_weight

            if weight_sum > 0:
                combined_scores[symbol] = combined / weight_sum

    if len(combined_scores) < 5:
        context['error'] = 'Insufficient data for multi-factor'
        return {}

    # Rank by combined score
    sorted_by_score = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

    n_stocks = len(sorted_by_score)
    n_top = max(1, int(n_stocks * top_pct))
    n_bottom = max(1, int(n_stocks * bottom_pct))

    long_stocks = [symbol for symbol, _ in sorted_by_score[:n_top]]
    short_stocks = [symbol for symbol, _ in sorted_by_score[-n_bottom:]] if allow_short else []

    orders = {}
    signals = {}

    # Position sizing
    total_positions = len(long_stocks) + len(short_stocks)

    if total_positions > 0:
        weight_per_position = 1.0 / total_positions  # Pure 1/N allocation

        # Long positions
        for symbol in long_stocks:
            orders[symbol] = {
                'action': 'target_weight',
                'weight': weight_per_position
            }
            signals[symbol] = f'QVM_LONG (score: {combined_scores[symbol]:.3f})'

        # Short positions
        for symbol in short_stocks:
            orders[symbol] = {
                'action': 'target_weight',
                'weight': -weight_per_position
            }
            signals[symbol] = f'QVM_SHORT (score: {combined_scores[symbol]:.3f})'

    # Close other positions
    active = long_stocks + short_stocks
    for symbol in market_data.keys():
        if symbol not in active:
            if symbol in positions and hasattr(positions[symbol], 'qty'):
                if positions[symbol].qty != 0:
                    orders[symbol] = {'action': 'close'}

    context['signals'] = signals
    context['combined_scores'] = combined_scores
    context['long_stocks'] = long_stocks
    context['short_stocks'] = short_stocks

    return orders
