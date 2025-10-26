# Machine Learning Adapter for Mean Reversion Trading Strategies

## Research Summary (2024-2025)

This document summarizes research on designing a simple machine learning adapter to enhance mean reversion trading strategies.

---

## 1. Core Concept

### Traditional vs ML-Enhanced Approach

**Traditional Mean Reversion:**
- Uses z-score: `(current_price - moving_average) / std_deviation`
- Fixed thresholds (e.g., buy when z-score < -2, sell when > 2)
- All signals treated equally

**ML-Enhanced Mean Reversion:**
- Uses classifier to predict **probability of bounce-back**
- Filters signals by confidence level (e.g., only trade when probability > 60%)
- Ranks opportunities by predicted reversion strength
- Adapts to market regime changes

### Key Insight
> "Rather than relying solely on traditional technical indicators, the model learns patterns from historical price action to identify high-probability mean reversion opportunities."

---

## 2. Simple ML Models for Mean Reversion

### Recommended Models (Simplest to Most Complex)

#### A. Logistic Regression (SIMPLEST)
**Best for:** Binary classification (will it revert: yes/no)

**Pros:**
- Extremely simple and interpretable
- Fast training and prediction
- Outputs probability directly
- Low risk of overfitting

**Use Case:** Predict if oversold stock will bounce within N days

**Implementation:**
```python
from sklearn.linear_model import LogisticRegression

# Train to predict: Will price revert to mean within 5 days?
model = LogisticRegression(C=1.0, max_iter=1000)
model.fit(X_train, y_train)  # y = 1 if reverted, 0 if not

# Get probability of reversion
prob_revert = model.predict_proba(X_current)[:, 1]
# Only trade if probability > 60%
if prob_revert > 0.60:
    enter_trade()
```

#### B. Random Forest Classifier (RECOMMENDED)
**Best for:** Handling non-linear relationships and feature interactions

**Pros:**
- Handles non-linear patterns well
- Provides feature importance scores
- Robust to noise
- Good default hyperparameters

**Implementation:**
```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,      # Number of trees
    max_depth=10,          # Prevent overfitting
    min_samples_split=50,  # Require meaningful patterns
    random_state=42
)
model.fit(X_train, y_train)

# Get probability and feature importance
prob_revert = model.predict_proba(X_current)[:, 1]
importance = model.feature_importances_
```

**Results from Research:**
- Average return: 15% per trade
- Can detect temporary price deviations in correlated assets

#### C. Recurrent Neural Networks (ADVANCED)
**Best for:** Capturing temporal patterns and sequences

**Pros:**
- Captures time-series patterns
- Learns long-term dependencies
- Can model price momentum that persists over time

**Cons:**
- More complex to implement
- Requires more data
- Longer training time
- Risk of overfitting

---

## 3. Feature Engineering for Mean Reversion

### Essential Features (Start with these)

#### Price-Based Features
```python
# 1. Z-Score (fundamental)
z_score = (close - sma_20) / std_20

# 2. Distance from moving averages
dist_sma_50 = (close - sma_50) / sma_50
dist_sma_200 = (close - sma_200) / sma_200

# 3. RSI (oversold/overbought)
rsi_14 = calculate_rsi(close, 14)

# 4. Bollinger Band position
bb_position = (close - bb_lower) / (bb_upper - bb_lower)
```

#### Return-Based Features
```python
# 5. Recent returns (capture momentum)
ret_1d = close.pct_change(1)
ret_5d = close.pct_change(5)
ret_20d = close.pct_change(20)

# 6. Volatility
vol_5d = returns.rolling(5).std()
vol_20d = returns.rolling(20).std()

# 7. Return normalization
(open - close) / open
(high - low) / low
```

#### Advanced Features
```python
# 8. QPI (Quantitative Price Index) - 3 day
# Identifies extreme oversold conditions below 15

# 9. Cross-sectional standardization
# Normalize indicators relative to broader market

# 10. Half-life of mean reversion
# Use as look-back window for rolling calculations
```

### Feature Importance (from research)
Top features for mean reversion:
1. **Z-score** (most important)
2. **Distance from 200-day SMA** (trend context)
3. **RSI** (momentum exhaustion)
4. **Short-term volatility** (instability signals reversion)
5. **Recent returns** (overextension)

---

## 4. Training Labels (Y variable)

### Binary Classification Approach

**Label Definition:**
```python
def create_labels(df, forward_days=5, reversion_threshold=0.02):
    """
    Label = 1 if price reverts by threshold% within forward_days
    Label = 0 otherwise

    Args:
        forward_days: How many days to check for reversion
        reversion_threshold: Minimum % return to consider reversion
    """

    # For oversold signals (buy)
    future_return = df['close'].shift(-forward_days) / df['close'] - 1

    # Label = 1 if price goes up by threshold%
    labels = (future_return > reversion_threshold).astype(int)

    return labels
```

**Example:**
- Stock is oversold (z-score = -2.5)
- Over next 5 days, price rises 3%
- Label = 1 (successful reversion)

### Probability-Based Filtering

After training, only trade when model confidence is high:

```python
# Get predicted probability
prob = model.predict_proba(features)[:, 1]

# Trading rules
if prob > 0.70:
    position_size = 1.0  # Full allocation
elif prob > 0.60:
    position_size = 0.5  # Half allocation
else:
    position_size = 0.0  # Skip trade
```

---

## 5. Simple ML Adapter Architecture

### Design Pattern

```
Traditional Signal → ML Filter → Enhanced Signal
     ↓                  ↓              ↓
  Bollinger         Probability    Trade if
  Band Touch    →   of Reversion  → prob > 60%
```

### Implementation Structure

```python
class MLMeanReversionAdapter:
    """
    ML adapter that wraps traditional mean reversion strategies
    """

    def __init__(self, base_strategy, model_type='random_forest'):
        self.base_strategy = base_strategy
        self.model = self._create_model(model_type)
        self.trained = False

    def train(self, historical_data):
        """Train ML model on historical signals"""

        # 1. Generate historical signals from base strategy
        signals = self.base_strategy.generate_signals(historical_data)

        # 2. Extract features at each signal point
        features = self._extract_features(historical_data, signals)

        # 3. Create labels (did price revert?)
        labels = self._create_labels(historical_data, signals)

        # 4. Train model
        self.model.fit(features, labels)
        self.trained = True

    def generate_signals(self, current_data):
        """Generate ML-filtered signals"""

        # 1. Get base strategy signals
        base_signals = self.base_strategy.generate_signals(current_data)

        # 2. Extract current features
        features = self._extract_features(current_data, base_signals)

        # 3. Predict probability of reversion
        probabilities = self.model.predict_proba(features)[:, 1]

        # 4. Filter by confidence threshold
        filtered_signals = []
        for signal, prob in zip(base_signals, probabilities):
            if prob > 0.60:  # Only high-confidence signals
                signal['ml_probability'] = prob
                signal['ml_confidence'] = 'high' if prob > 0.70 else 'medium'
                filtered_signals.append(signal)

        return filtered_signals
```

---

## 6. Training Data Best Practices

### Data Integrity (Critical!)

**From Research:**
> "Models trained exclusively on Russell 3000 constituents during periods when stocks were index members. Training excludes pre-index and post-delisted periods to avoid 'confusing the model' with irrelevant patterns."

**Key Rules:**
1. **Survivorship Bias:** Only use data from when stock was actively traded
2. **Look-ahead Bias:** Never use future information in features
3. **Train/Test Split:** Use time-based splits (e.g., train on 2015-2020, test on 2021-2023)
4. **Rolling Window:** Retrain model periodically (e.g., every 6 months)

### Sample Size Requirements

**Minimum Recommendations:**
- **Logistic Regression:** 500+ labeled examples
- **Random Forest:** 1,000+ labeled examples
- **Neural Networks:** 10,000+ labeled examples

---

## 7. Regime Filtering (Advanced)

### Realized Volatility-Based Filter (General Approach)

**Concept:**
Realized volatility measures actual price variability and can be calculated for ANY asset (not just S&P 500 like VIX). Mean reversion strategies perform best during high volatility regimes when prices overextend and snap back.

**From Research:**
> "Volatility-based filter reduces maximum drawdown from 50% to 19%"

**Why Realized Volatility > VIX:**

| Feature | VIX | Realized Volatility |
|---------|-----|---------------------|
| **Generality** | Only S&P 500 | Any asset |
| **Availability** | Requires separate data | Calculated from price |
| **Forward-looking** | Yes (implied vol) | No (historical) |
| **Implementation** | Need API/data source | Built-in calculation |
| **Cost** | Data subscription | Free |
| **Applicability** | US equities mainly | Stocks, crypto, forex, commodities |

**Basic Implementation:**
```python
import numpy as np

def calculate_realized_volatility(returns, window=20):
    """
    Calculate realized volatility from returns

    Args:
        returns: Daily returns series
        window: Rolling window (default 20 days)

    Returns:
        Annualized realized volatility
    """
    # Standard deviation of returns over window
    realized_vol = returns.rolling(window).std() * np.sqrt(252)
    return realized_vol

def should_trade_mean_reversion(current_vol, historical_vol, lookback=252):
    """
    Determine if market regime favors mean reversion

    Args:
        current_vol: Current realized volatility
        historical_vol: Historical volatility series
        lookback: Days to look back for percentile calculation

    Returns:
        (should_trade, regime_label, position_scaler)
    """

    # Calculate volatility percentile over lookback period
    recent_vol = historical_vol.tail(lookback)
    vol_percentile = (current_vol - recent_vol.min()) / (recent_vol.max() - recent_vol.min())

    # Regime classification
    # - High Vol (>80th percentile): Mean reversion WORKS BEST (panic/euphoria)
    # - Medium Vol (20-80th): Normal conditions, use ML filter
    # - Low Vol (<20th): AVOID mean reversion (trending/quiet markets)

    if vol_percentile > 0.80:
        return True, 'high_volatility', 1.5  # Increase position size 50%
    elif vol_percentile > 0.60:
        return True, 'elevated_volatility', 1.0  # Normal position size
    elif vol_percentile > 0.20:
        return True, 'normal_volatility', 0.75  # Reduce position size
    else:
        return False, 'low_volatility', 0.0  # Skip mean reversion trades
```

**Alternative: Z-Score Approach (More Robust)**
```python
def volatility_regime_zscore(current_vol, historical_vol, window=252):
    """
    Use z-score to classify volatility regime
    More statistically robust than percentiles
    """
    vol_mean = historical_vol.tail(window).mean()
    vol_std = historical_vol.tail(window).std()

    vol_zscore = (current_vol - vol_mean) / vol_std

    if vol_zscore > 2.0:
        return True, 'extreme_high_vol', 1.5
    elif vol_zscore > 1.0:
        return True, 'high_vol', 1.25
    elif vol_zscore > -0.5:
        return True, 'normal_vol', 1.0
    elif vol_zscore > -1.5:
        return True, 'low_vol', 0.5
    else:
        return False, 'extreme_low_vol', 0.0
```

**Advanced Volatility Estimators:**

```python
# 1. Parkinson's Volatility (uses High-Low range - more efficient)
def parkinson_volatility(high, low, window=20):
    """
    Parkinson's volatility estimator
    More efficient than close-to-close volatility
    Uses high-low range information
    """
    hl_ratio = np.log(high / low) ** 2
    parkinson_vol = np.sqrt(hl_ratio.rolling(window).mean() / (4 * np.log(2))) * np.sqrt(252)
    return parkinson_vol

# 2. Garman-Klass Volatility (uses OHLC - most efficient)
def garman_klass_volatility(open_price, high, low, close, window=20):
    """
    Garman-Klass volatility estimator
    Most efficient unbiased estimator using OHLC data
    """
    term1 = 0.5 * (np.log(high / low) ** 2)
    term2 = -(2 * np.log(2) - 1) * (np.log(close / open_price) ** 2)

    gk_vol = np.sqrt((term1 - term2).rolling(window).mean()) * np.sqrt(252)
    return gk_vol
```

**Practical Usage Example:**

```python
def enhanced_mean_reversion_with_vol_filter(prices):
    """
    Mean reversion strategy with realized volatility regime filter
    Works for ANY asset - stocks, crypto, forex, commodities
    """

    # Calculate returns and realized volatility
    returns = prices.pct_change()
    realized_vol = calculate_realized_volatility(returns, window=20)

    # Get current volatility regime
    current_vol = realized_vol.iloc[-1]
    should_trade, regime, position_scaler = should_trade_mean_reversion(
        current_vol,
        realized_vol,
        lookback=252
    )

    if not should_trade:
        return None  # Skip mean reversion in low vol regime

    # Generate base mean reversion signal
    z_score = calculate_zscore(prices)

    if z_score < -2.0:  # Oversold
        # Scale position by volatility regime
        base_size = 1.0
        adjusted_size = base_size * position_scaler

        return {
            'action': 'buy',
            'size': adjusted_size,
            'regime': regime,
            'vol_percentile': (current_vol - realized_vol.min()) / (realized_vol.max() - realized_vol.min()),
            'reason': f'Mean reversion buy in {regime} (vol={current_vol:.2%})'
        }

    return None
```

**Results:**
- Mean reversion performs best in high volatility regimes (>80th percentile)
- Low volatility favors trend-following, not mean reversion
- Dynamic position sizing by regime improves risk-adjusted returns
- Works for any asset without requiring external volatility indices

---

## 8. Performance Metrics (from Research)

### ML-Enhanced Results

**Traditional Bollinger Bands Strategy:**
- Annual Return: ~10-12%
- Sharpe Ratio: ~0.5
- Max Drawdown: ~30%

**ML-Enhanced Strategy:**
- Annual Return: **41.9%**
- Sharpe Ratio: **1.55**
- Max Drawdown: **19%** (with volatility regime filter)

**Key Improvements:**
- Higher probability filtering (only >60% confidence)
- Dynamic position sizing (ranked by prediction strength)
- Regime awareness (realized volatility-based filtering)

---

## 9. Practical Implementation Steps

### Step 1: Start Simple (Logistic Regression)

```python
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

# 1. Prepare data
def prepare_training_data(prices, signals):
    """Extract features at each signal point"""

    features = pd.DataFrame()

    # Add features
    features['z_score'] = (prices - prices.rolling(20).mean()) / prices.rolling(20).std()
    features['rsi'] = calculate_rsi(prices, 14)
    features['dist_sma_200'] = (prices - prices.rolling(200).mean()) / prices.rolling(200).mean()
    features['vol_20'] = prices.pct_change().rolling(20).std()
    features['ret_5d'] = prices.pct_change(5)

    return features

# 2. Create labels
def create_labels(prices, forward_days=5):
    """Binary: did price revert?"""
    future_return = prices.shift(-forward_days) / prices - 1
    labels = (future_return > 0.02).astype(int)  # 2% threshold
    return labels

# 3. Train model
features = prepare_training_data(prices, signals)
labels = create_labels(prices)

model = LogisticRegression(C=1.0, max_iter=1000)
model.fit(features, labels)

# 4. Make predictions
prob_revert = model.predict_proba(current_features)[:, 1]
```

### Step 2: Add Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=50,
    random_state=42
)

model.fit(features, labels)

# Check feature importance
importance = pd.DataFrame({
    'feature': features.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(importance)
```

### Step 3: Backtest with Threshold

```python
# Only trade high-confidence signals
for signal in base_signals:
    prob = model.predict_proba([signal.features])[:, 1]

    if prob > 0.70:
        # High confidence - full position
        execute_trade(signal, size=1.0)
    elif prob > 0.60:
        # Medium confidence - half position
        execute_trade(signal, size=0.5)
    else:
        # Skip - low confidence
        pass
```

---

## 10. Common Pitfalls to Avoid

1. **Overfitting:** Using too many features or too complex models
   - Solution: Start simple, use cross-validation

2. **Look-ahead Bias:** Using future information in features
   - Solution: Strict time-based train/test splits

3. **Data Snooping:** Training and testing on same period
   - Solution: Always use out-of-sample testing

4. **Ignoring Regime Changes:** Model trained on trending market used in ranging market
   - Solution: Add VIX filter or retrain regularly

5. **Not Enough Data:** Training complex model on small dataset
   - Solution: Start with logistic regression, need 500+ examples minimum

---

## 11. Recommended Starting Point

### Minimal Viable Implementation

```python
# Simple ML adapter for existing Bollinger Bands strategy

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit

class SimplMLAdapter:
    """
    Wraps any mean reversion strategy with ML filtering
    """

    def __init__(self, confidence_threshold=0.60):
        self.model = LogisticRegression(max_iter=1000)
        self.threshold = confidence_threshold
        self.trained = False

    def train(self, historical_prices, historical_signals):
        """
        Train on historical signals

        Args:
            historical_prices: DataFrame with OHLCV data
            historical_signals: List of timestamps where strategy signaled
        """

        # Extract features at signal points
        X = []
        y = []

        for signal_date in historical_signals:
            # Features
            z_score = self._calculate_zscore(historical_prices, signal_date)
            rsi = self._calculate_rsi(historical_prices, signal_date)
            dist_sma = self._calculate_dist_sma(historical_prices, signal_date)

            features = [z_score, rsi, dist_sma]
            X.append(features)

            # Label: did price revert in next 5 days?
            future_price = historical_prices.loc[signal_date + timedelta(days=5), 'close']
            current_price = historical_prices.loc[signal_date, 'close']
            reverted = (future_price / current_price - 1) > 0.02

            y.append(1 if reverted else 0)

        # Train
        self.model.fit(X, y)
        self.trained = True

    def filter_signal(self, current_features):
        """
        Returns: (should_trade, probability)
        """
        prob = self.model.predict_proba([current_features])[0, 1]
        should_trade = prob > self.threshold
        return should_trade, prob
```

### Usage with Existing Strategy

```python
# 1. Create adapter
ml_adapter = SimplMLAdapter(confidence_threshold=0.60)

# 2. Train on historical data
ml_adapter.train(historical_prices, historical_signals)

# 3. Use in live trading
def enhanced_strategy(prices, current_time):
    # Get base signal
    if bollinger_bands_signal(prices, current_time):

        # Extract features
        features = [
            calculate_zscore(prices, current_time),
            calculate_rsi(prices, current_time),
            calculate_dist_sma(prices, current_time)
        ]

        # ML filter
        should_trade, prob = ml_adapter.filter_signal(features)

        if should_trade:
            return {
                'action': 'buy',
                'ml_probability': prob,
                'ml_filtered': True
            }

    return None
```

---

## 12. Resources & References

### Key Papers
- "Modeling Investor Behavior Using Machine Learning: Mean-Reversion and Momentum Trading Strategies" (2019)
- "Passive Aggressive Mean Reversion Strategy for Portfolio Selection" (Machine Learning Journal)

### Tutorials
- Medium: "Using Random Forests to Improve Mean Reversion Strategy"
- QuantInsti: "Random Forest Algorithm in Trading Using Python"
- GitHub: "Simple Mean Reversion Strategy in Python" (arendarski)

### Tools
- **scikit-learn**: LogisticRegression, RandomForestClassifier
- **pandas**: Feature engineering and data manipulation
- **yfinance**: Historical price data
- **backtesting.py**: Strategy backtesting framework

---

## 13. Next Steps for Implementation

1. **Week 1:** Implement logistic regression adapter with 3-5 features
2. **Week 2:** Backtest on historical data, tune confidence threshold
3. **Week 3:** Add random forest, compare results
4. **Week 4:** Add realized volatility regime filter
5. **Week 5:** Paper trade with live data

### Success Criteria
- Model accuracy > 55% on test set
- Sharpe ratio improvement > 0.3 vs base strategy
- Max drawdown reduction > 10%
- Computational efficiency < 100ms per prediction

---

## Conclusion

**Key Takeaways:**

1. **Start Simple:** Logistic regression with 5 features is often enough
2. **Filter, Don't Replace:** Use ML to filter traditional signals, not replace them
3. **Focus on Probability:** Trade only high-confidence signals (>60%)
4. **Mind the Data:** Avoid look-ahead bias and survivorship bias
5. **Add Regime Filter:** Realized volatility-based filtering dramatically improves results

**Expected Improvements:**
- Win rate: +10-15%
- Sharpe ratio: +0.5-1.0
- Max drawdown: -10-15%

The ML adapter should be viewed as a **signal quality filter**, not a complete replacement for domain knowledge and traditional technical analysis.
