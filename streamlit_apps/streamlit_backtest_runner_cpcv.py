"""
CPCV Validation Sheet for Streamlit Backtest Runner

This module adds professional CPCV validation capabilities to the Streamlit interface.
Insert this code into streamlit_backtest_runner.py before the main() function.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def render_cpcv_validation_sheet():
    """Sheet 4: CPCV Validation - Professional Strategy Validation"""
    st.subheader("🔬 CPCV Validation")
    st.caption("**Combinatorial Purged Cross-Validation** - Detect overfitting and validate strategy robustness")

    if 'config' not in st.session_state or 'selected_strategy_name' not in st.session_state:
        st.warning("Please configure backtest parameters and select a strategy first.")
        st.info("👈 Go to 'Configuration' and 'Strategy' tabs to set up your backtest.")
        return

    # CPCV Mode Selection
    st.write("")
    mode = st.radio(
        "Validation Mode",
        ["Single Strategy Validation", "Parameter Grid Optimization", "Strategy Comparison"],
        horizontal=True,
        label_visibility="collapsed"
    )

    st.write("")  # Spacer

    if mode == "Single Strategy Validation":
        render_single_strategy_cpcv()
    elif mode == "Parameter Grid Optimization":
        render_parameter_optimization_cpcv()
    else:
        render_strategy_comparison_cpcv()


def render_single_strategy_cpcv():
    """Validate a single strategy with CPCV"""
    st.caption("**Single Strategy Validation**")
    st.write("Validate one strategy configuration across multiple train/test paths to detect overfitting.")

    # CPCV Configuration
    st.write("")
    st.caption("**CPCV Settings**")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        n_splits = st.number_input("Number of Folds", value=10, min_value=3, max_value=20, step=1)
    with col2:
        n_test_splits = st.number_input("Test Folds per Path", value=2, min_value=1, max_value=5, step=1)
    with col3:
        purge_pct = st.number_input("Purge %", value=5.0, min_value=0.0, max_value=20.0, step=1.0) / 100
    with col4:
        embargo_pct = st.number_input("Embargo %", value=2.0, min_value=0.0, max_value=10.0, step=1.0) / 100

    # Calculate number of paths
    import math
    n_paths = math.comb(n_splits, n_test_splits)
    st.caption(f"This will generate **{n_paths} validation paths** (C({n_splits},{n_test_splits}))")

    # Run Button
    st.write("")
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        run_validation = st.button("🚀 Run CPCV Validation", type="primary", use_container_width=True)

    if run_validation:
        with st.spinner(f"Running CPCV validation across {n_paths} paths... This may take a few minutes."):
            # Get configuration from session
            config = st.session_state.config
            strategy_name = st.session_state.selected_strategy_name
            strategy_params = st.session_state.get('strategy_params', {})

            # Get strategy function
            strategies = get_available_strategies()
            strategy_func = strategies[strategy_name]['function']

            # Create CPCV config
            cpcv_config = CPCVConfig(
                n_splits=n_splits,
                n_test_splits=n_test_splits,
                purge_pct=purge_pct,
                embargo_pct=embargo_pct
            )

            # Run CPCV validation
            try:
                validator = CPCVValidator(config, cpcv_config)
                result = validator.validate(
                    strategy=strategy_func,
                    symbols=config.universe if hasattr(config, 'universe') else ['SPY'],
                    strategy_params=strategy_params
                )

                # Store result in session
                st.session_state.cpcv_result = result

                st.success(f"Validation complete! Ran {result.n_paths} paths in {result.total_runtime_seconds:.1f}s")

            except Exception as e:
                st.error(f"CPCV validation failed: {str(e)}")
                st.exception(e)
                return

    # Display results if available
    if 'cpcv_result' in st.session_state:
        display_cpcv_results(st.session_state.cpcv_result)


def display_cpcv_results(result):
    """Display comprehensive CPCV validation results"""
    st.write("")
    st.caption("**📊 Validation Results**")

    # Summary Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Paths Completed", f"{result.n_paths}")
    with col2:
        st.metric("Mean Sharpe", f"{result.mean_sharpe:.3f}")
    with col3:
        st.metric("Std Sharpe", f"{result.std_sharpe:.3f}")
    with col4:
        st.metric("Mean Return", f"{result.mean_return:.1%}")
    with col5:
        st.metric("Mean Max DD", f"{result.mean_max_drawdown:.1%}")

    st.write("")

    # Overfitting Metrics - Highlight Box
    st.caption("**🎯 Overfitting Detection Metrics**")

    metrics = result.overfitting_metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        pbo_color = "green" if metrics.pbo < 0.3 else ("orange" if metrics.pbo < 0.5 else "red")
        st.markdown(f"""
        <div style='background-color: rgba(0,255,0,0.1) if '{pbo_color}' == 'green' else rgba(255,165,0,0.1) if '{pbo_color}' == 'orange' else rgba(255,0,0,0.1);
                    padding: 0.8rem; border-radius: 0.5rem; border-left: 4px solid {pbo_color};'>
            <p style='margin: 0; font-size: 0.75rem; color: #666;'>PBO</p>
            <p style='margin: 0; font-size: 1.5rem; font-weight: bold;'>{metrics.pbo:.1%}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        dsr_color = "green" if metrics.dsr > 2.0 else ("orange" if metrics.dsr > 1.0 else "red")
        st.markdown(f"""
        <div style='background-color: rgba(0,255,0,0.1) if '{dsr_color}' == 'green' else rgba(255,165,0,0.1) if '{dsr_color}' == 'orange' else rgba(255,0,0,0.1);
                    padding: 0.8rem; border-radius: 0.5rem; border-left: 4px solid {dsr_color};'>
            <p style='margin: 0; font-size: 0.75rem; color: #666;'>DSR</p>
            <p style='margin: 0; font-size: 1.5rem; font-weight: bold;'>{metrics.dsr:.2f}</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        deg_color = "green" if abs(metrics.degradation_pct) < 10 else ("orange" if abs(metrics.degradation_pct) < 20 else "red")
        st.markdown(f"""
        <div style='background-color: rgba(0,255,0,0.1) if '{deg_color}' == 'green' else rgba(255,165,0,0.1) if '{deg_color}' == 'orange' else rgba(255,0,0,0.1);
                    padding: 0.8rem; border-radius: 0.5rem; border-left: 4px solid {deg_color};'>
            <p style='margin: 0; font-size: 0.75rem; color: #666;'>Degradation</p>
            <p style='margin: 0; font-size: 1.5rem; font-weight: bold;'>{metrics.degradation_pct:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        stab_color = "green" if metrics.sharpe_stability > 5.0 else ("orange" if metrics.sharpe_stability > 2.0 else "red")
        st.markdown(f"""
        <div style='background-color: rgba(0,255,0,0.1) if '{stab_color}' == 'green' else rgba(255,165,0,0.1) if '{stab_color}' == 'orange' else rgba(255,0,0,0.1);
                    padding: 0.8rem; border-radius: 0.5rem; border-left: 4px solid {stab_color};'>
            <p style='margin: 0; font-size: 0.75rem; color: #666;'>Stability</p>
            <p style='margin: 0; font-size: 1.5rem; font-weight: bold;'>{metrics.sharpe_stability:.2f}</p>
        </div>
        """, unsafe_allow_html=True)

    # Interpretations
    st.write("")
    with st.expander("📖 Metric Interpretations", expanded=True):
        for metric, interpretation in result.overfitting_interpretations.items():
            st.write(f"• **{metric.upper()}:** {interpretation}")

    # Validation Status
    st.write("")
    if result.passes_validation():
        st.success("✅ **Strategy PASSES validation criteria**")
    else:
        st.error("❌ **Strategy FAILS validation criteria**")
        if result.validation_warnings:
            for warning in result.validation_warnings:
                st.warning(f"⚠️ {warning}")

    # Visualization Section
    st.write("")
    st.caption("**📈 Validation Path Analysis**")

    # Create path distribution chart
    fig = create_path_distribution_chart(result)
    st.plotly_chart(fig, use_container_width=True)

    # Performance distribution
    col1, col2 = st.columns(2)
    with col1:
        fig_dist = create_sharpe_distribution_chart(result)
        st.plotly_chart(fig_dist, use_container_width=True)

    with col2:
        fig_scatter = create_path_scatter_chart(result)
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Detailed path results table
    st.write("")
    with st.expander("🔍 Detailed Path Results", expanded=False):
        path_data = []
        for path in result.path_results:
            path_data.append({
                'Path ID': path.path_id,
                'Test Folds': str(path.test_fold_indices),
                'Sharpe Ratio': f"{path.sharpe_ratio:.3f}",
                'Return': f"{path.total_return:.2%}",
                'Max Drawdown': f"{path.max_drawdown:.2%}"
            })

        path_df = pd.DataFrame(path_data)
        st.dataframe(path_df, use_container_width=True, hide_index=True)


def create_path_distribution_chart(result):
    """Create chart showing Sharpe ratio across all validation paths"""
    sharpe_values = [p.sharpe_ratio for p in result.path_results]
    path_ids = [p.path_id for p in result.path_results]

    fig = go.Figure()

    # Bar chart of Sharpe ratios
    fig.add_trace(go.Bar(
        x=path_ids,
        y=sharpe_values,
        name='Sharpe Ratio',
        marker=dict(
            color=sharpe_values,
            colorscale='RdYlGn',
            cmin=-2,
            cmax=2,
            colorbar=dict(title="Sharpe")
        )
    ))

    # Add mean line
    fig.add_hline(
        y=result.mean_sharpe,
        line_dash="dash",
        line_color="blue",
        annotation_text=f"Mean: {result.mean_sharpe:.3f}"
    )

    fig.update_layout(
        title="Sharpe Ratio Across All Validation Paths",
        xaxis_title="Path ID",
        yaxis_title="Sharpe Ratio",
        height=350,
        hovermode='x',
        showlegend=False
    )

    return fig


def create_sharpe_distribution_chart(result):
    """Create histogram of Sharpe ratio distribution"""
    sharpe_values = [p.sharpe_ratio for p in result.path_results]

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=sharpe_values,
        nbinsx=20,
        name='Distribution',
        marker=dict(color='steelblue')
    ))

    # Add mean line
    fig.add_vline(
        x=result.mean_sharpe,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {result.mean_sharpe:.3f}"
    )

    fig.update_layout(
        title="Sharpe Ratio Distribution",
        xaxis_title="Sharpe Ratio",
        yaxis_title="Frequency",
        height=300,
        showlegend=False
    )

    return fig


def create_path_scatter_chart(result):
    """Create scatter plot of Return vs Drawdown"""
    returns = [p.total_return * 100 for p in result.path_results]
    drawdowns = [abs(p.max_drawdown * 100) for p in result.path_results]
    sharpe = [p.sharpe_ratio for p in result.path_results]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=drawdowns,
        y=returns,
        mode='markers',
        marker=dict(
            size=10,
            color=sharpe,
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="Sharpe"),
            cmin=-2,
            cmax=2
        ),
        text=[f"Path {p.path_id}" for p in result.path_results],
        hovertemplate='<b>%{text}</b><br>Return: %{y:.1f}%<br>Max DD: %{x:.1f}%'
    ))

    fig.update_layout(
        title="Return vs Max Drawdown (colored by Sharpe)",
        xaxis_title="Max Drawdown (%)",
        yaxis_title="Total Return (%)",
        height=300
    )

    return fig


def render_parameter_optimization_cpcv():
    """Parameter grid optimization with CPCV validation"""
    st.caption("**Parameter Grid Optimization with CPCV**")
    st.write("Optimize strategy parameters using parallel search, then validate top candidates with CPCV.")

    # Get configuration from session
    if 'config' not in st.session_state or 'selected_strategy_name' not in st.session_state:
        st.warning("Please configure backtest parameters and select a strategy first.")
        return

    config = st.session_state.config
    strategy_name = st.session_state.selected_strategy_name

    # Get strategy function
    strategies = get_available_strategies()
    strategy_func = strategies[strategy_name]['function']

    st.write("")
    st.caption("**Step 1: Choose Optimization Method**")

    col1, col2 = st.columns(2)
    with col1:
        optimization_method = st.selectbox(
            "Optimization Algorithm",
            ["Grid Search (Exhaustive)", "FLAML (Intelligent)"],
            help="Grid Search tests all combinations. FLAML intelligently explores the space (10x faster)."
        )

    with col2:
        optimization_metric = st.selectbox(
            "Optimization Metric",
            ["sharpe_ratio", "sortino_ratio", "total_return", "calmar_ratio"],
            help="Metric to maximize during parameter search"
        )

    st.write("")
    st.caption("**Step 2: Define Parameter Space**")

    # Extract parameters from strategy
    from streamlit_backtest_runner import extract_strategy_params
    strategy_params = extract_strategy_params(strategy_func)

    if strategy_params:
        st.write("**Strategy Parameters Detected:**")

        # Show available parameters as buttons
        cols = st.columns(min(len(strategy_params), 4))
        for idx, (param_name, param_info) in enumerate(strategy_params.items()):
            with cols[idx % len(cols)]:
                default_val = param_info.get('default', 'N/A')
                if st.button(f"➕ {param_name}", key=f"add_param_{param_name}",
                            help=f"Type: {param_info.get('type', 'unknown')}, Default: {default_val}"):
                    if 'param_definitions' not in st.session_state:
                        st.session_state.param_definitions = []
                    if param_name not in [p['name'] for p in st.session_state.param_definitions]:
                        # Suggest ranges based on parameter type and default
                        suggested_min = 5
                        suggested_max = 50
                        suggested_step = 5

                        if default_val and default_val != 'N/A':
                            try:
                                default_num = float(default_val)
                                if param_info.get('type') == 'int':
                                    suggested_min = max(1, int(default_num * 0.5))
                                    suggested_max = int(default_num * 2)
                                    suggested_step = max(1, int(default_num * 0.1))
                                elif param_info.get('type') == 'float':
                                    suggested_min = round(default_num * 0.5, 4)
                                    suggested_max = round(default_num * 2, 4)
                                    suggested_step = round(default_num * 0.1, 4)
                            except:
                                pass

                        st.session_state.param_definitions.append({
                            'name': param_name,
                            'type': param_info.get('type', 'int'),
                            'default': default_val,
                            'min': suggested_min,
                            'max': suggested_max,
                            'step': suggested_step
                        })
                        st.rerun()

    st.write("")
    st.write("**Or add custom parameter:**")

    if 'param_definitions' not in st.session_state:
        st.session_state.param_definitions = []

    col1, col2 = st.columns([3, 1])
    with col1:
        new_param_name = st.text_input("Parameter Name", key="new_param", placeholder="e.g., custom_param")
    with col2:
        if st.button("➕ Add Custom"):
            if new_param_name and new_param_name not in [p['name'] for p in st.session_state.param_definitions]:
                st.session_state.param_definitions.append({
                    'name': new_param_name,
                    'type': 'int',
                    'min': 5,
                    'max': 50,
                    'step': 5
                })
                st.rerun()

    # Display parameter configurations
    param_grid = {}
    if st.session_state.param_definitions:
        st.write("")
        st.write("**Configure Parameter Ranges:**")

        for idx, param_def in enumerate(st.session_state.param_definitions):
            col1, col2, col3, col4, col5, col6 = st.columns([2, 1, 1, 1, 1, 0.5])

            with col1:
                param_type = param_def.get('type', 'int')
                default_val = param_def.get('default', 'N/A')
                st.write(f"**{param_def['name']}**")
                if default_val != 'N/A':
                    st.caption(f"Default: {default_val}")

            # Determine step format based on type
            is_float = param_type == 'float'

            if is_float:
                # Float parameters
                min_default = float(param_def.get('min', 0.01))
                max_default = float(param_def.get('max', 1.0))
                step_default = float(param_def.get('step', 0.01))
                step_format = 0.001
            else:
                # Integer parameters
                min_default = int(param_def.get('min', 5))
                max_default = int(param_def.get('max', 50))
                step_default = int(param_def.get('step', 5))
                step_format = 1

            with col2:
                min_val = st.number_input(
                    f"Min",
                    key=f"min_{idx}",
                    value=min_default,
                    step=step_format,
                    format="%.4f" if is_float else "%d"
                )

            with col3:
                max_val = st.number_input(
                    f"Max",
                    key=f"max_{idx}",
                    value=max_default,
                    step=step_format,
                    format="%.4f" if is_float else "%d"
                )

            with col4:
                step_val = st.number_input(
                    f"Step",
                    key=f"step_{idx}",
                    value=step_default,
                    step=step_format,
                    min_value=step_format,
                    format="%.4f" if is_float else "%d"
                )

            with col5:
                # Show number of values
                if step_val > 0:
                    n_values = int((max_val - min_val) / step_val) + 1
                    st.caption(f"({n_values} values)")
                else:
                    st.caption("⚠️")

            with col6:
                if st.button("🗑️", key=f"del_{idx}"):
                    st.session_state.param_definitions.pop(idx)
                    st.rerun()

            # Build parameter grid
            if is_float:
                # For float parameters, create list with proper precision
                param_grid[param_def['name']] = list(np.arange(min_val, max_val + step_val, step_val))
            else:
                # For int parameters, use range
                param_grid[param_def['name']] = list(range(int(min_val), int(max_val) + 1, int(step_val)))

        # Show parameter space size
        if param_grid:
            total_combinations = 1
            for values in param_grid.values():
                total_combinations *= len(values)

            st.write("")
            if optimization_method == "Grid Search (Exhaustive)":
                st.info(f"📊 **{total_combinations} total combinations** will be tested (exhaustive)")
            else:
                est_evals = min(total_combinations, 100)
                st.info(f"🧠 **~{est_evals} combinations** will be intelligently sampled from {total_combinations} possible (FLAML)")

    # FLAML-specific settings
    if optimization_method == "FLAML (Intelligent)":
        st.write("")
        st.caption("**FLAML Settings**")
        col1, col2 = st.columns(2)
        with col1:
            time_budget_s = st.number_input("Time Budget (seconds)", value=300, min_value=10, max_value=3600,
                                          help="Maximum time for FLAML search")
        with col2:
            num_samples = st.number_input("Max Samples", value=-1, min_value=-1, max_value=1000,
                                        help="Max evaluations (-1 = unlimited)")

    st.write("")
    st.caption("**Step 3: CPCV Validation Settings**")

    col1, col2 = st.columns(2)
    with col1:
        top_k = st.number_input("Top K to Validate", value=3, min_value=1, max_value=10,
                               help="Number of best parameter sets to validate with CPCV")
    with col2:
        n_splits = st.number_input("CPCV Folds", value=10, min_value=3, max_value=20)

    col1, col2, col3 = st.columns(3)
    with col1:
        n_test_splits = st.number_input("Test Folds", value=2, min_value=1, max_value=5)
    with col2:
        purge_pct = st.number_input("Purge %", value=5.0, min_value=0.0, max_value=20.0) / 100
    with col3:
        embargo_pct = st.number_input("Embargo %", value=2.0, min_value=0.0, max_value=10.0) / 100

    # Run Button
    st.write("")
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        run_optimization = st.button("🚀 Run Optimization", type="primary", use_container_width=True)

    if run_optimization and param_grid:
        from backt.optimization.optimizer import StrategyOptimizer
        from backt.validation.cpcv_validator import CPCVConfig
        from backt import BacktestConfig

        try:
            # Convert config dict to BacktestConfig object if needed
            if isinstance(config, dict):
                from backt.utils.config import ExecutionConfig

                # Extract symbols separately (not a BacktestConfig parameter)
                symbols = config.get('symbols', ['SPY'])

                # Extract execution-related parameters
                execution_config = ExecutionConfig(
                    spread=config.get('spread', 0.01),
                    slippage_pct=config.get('slippage_pct', 0.0005),
                    commission_per_share=config.get('commission_per_share', 0.001)
                )

                # Create BacktestConfig with only valid parameters
                backtest_config = BacktestConfig(
                    start_date=config['start_date'],
                    end_date=config['end_date'],
                    initial_capital=config.get('initial_capital', 100000.0),
                    allow_short=config.get('allow_short', False),
                    max_leverage=config.get('max_leverage', 1.0),
                    max_position_size=config.get('max_position_size', None),
                    execution=execution_config
                )
            else:
                backtest_config = config
                symbols = backtest_config.universe if hasattr(backtest_config, 'universe') else ['SPY']

            # Show optimization method
            with st.spinner(f"Running {optimization_method}..."):
                # Create CPCV config
                cpcv_config = CPCVConfig(
                    n_splits=n_splits,
                    n_test_splits=n_test_splits,
                    purge_pct=purge_pct,
                    embargo_pct=embargo_pct,
                    n_jobs=-1,  # Use all CPU cores
                    use_numba=True
                )

                # Branch based on optimization method
                if optimization_method == "FLAML (Intelligent)":
                    from backt.optimization.flaml_optimizer import FLAMLOptimizer
                    from flaml import tune

                    # Convert param_grid to FLAML format
                    flaml_param_space = {}
                    for param_name, param_values in param_grid.items():
                        # Determine if parameter is int or float
                        if all(isinstance(v, int) for v in param_values):
                            # Integer parameter
                            min_val = min(param_values)
                            max_val = max(param_values)
                            flaml_param_space[param_name] = {'domain': tune.randint(min_val, max_val + 1)}
                        else:
                            # Float parameter
                            min_val = min(param_values)
                            max_val = max(param_values)
                            flaml_param_space[param_name] = {'domain': tune.uniform(min_val, max_val)}

                    # Create FLAML optimizer
                    optimizer = FLAMLOptimizer(
                        strategy_function=strategy_func,
                        config=backtest_config,
                        symbols=symbols
                    )

                    # Run FLAML optimization with CPCV
                    result = optimizer.optimize_with_cpcv(
                        param_space=flaml_param_space,
                        optimization_metric=optimization_metric,
                        minimize=False,  # We want to maximize metrics like Sharpe
                        time_budget_s=time_budget_s,
                        num_samples=num_samples,
                        top_k=top_k,
                        cpcv_config=cpcv_config,
                        verbose=3
                    )

                else:  # Grid Search
                    # Create grid search optimizer
                    optimizer = StrategyOptimizer(
                        strategy_function=strategy_func,
                        config=backtest_config,
                        symbols=symbols
                    )

                    # Run optimization with CPCV
                    result = optimizer.optimize_with_cpcv(
                        param_grid=param_grid,
                        optimization_metric=optimization_metric,
                        n_jobs=-1,  # Parallel processing
                        top_k=top_k,
                        cpcv_config=cpcv_config,
                        verbose=False
                    )

                # Store results in session
                st.session_state.optimization_result = result

            # Handle different result formats (FLAML vs Grid Search)
            if hasattr(result, 'total_evaluations'):
                # FLAML result format (from results.py)
                n_evals = result.total_evaluations
                exec_time = result.total_time_seconds
                best_params = result.best_parameters
                best_value = result.best_metrics.get(optimization_metric, np.nan)
            else:
                # Grid search result format (from optimizer.py)
                n_evals = result.total_combinations
                exec_time = result.execution_time
                best_params = result.best_params
                best_value = result.best_metric_value

            st.success(f"✅ Optimization complete! Tested {n_evals} combinations in {exec_time:.1f}s")

            # Display results
            st.write("")
            st.caption("**Optimization Results**")

            # Top parameters
            st.write("**Best Parameters:**")
            st.json(best_params)

            st.write(f"**Best {optimization_metric}:** {best_value:.4f}")

            # Top K results
            st.write("")
            st.caption(f"**Top {top_k} Parameter Sets**")

            # Convert results to DataFrame if needed (for FLAML)
            if hasattr(result, 'total_evaluations'):
                # FLAML format: List[ParameterSetResult]
                top_results = result.get_top_k(top_k)
                rows = []
                for param_result in top_results:
                    row = {}
                    # Add parameters
                    for param_name, param_value in param_result.parameters.items():
                        row[f'param_{param_name}'] = param_value
                    # Add metrics
                    row.update(param_result.metrics)
                    rows.append(row)
                top_df = pd.DataFrame(rows)
            else:
                # Grid search format: DataFrame
                top_df = result.all_results.head(top_k).copy()

            param_cols = [col for col in top_df.columns if col.startswith('param_')]
            metric_cols = [optimization_metric, 'total_return', 'max_drawdown', 'sharpe_ratio']
            metric_cols = [col for col in metric_cols if col in top_df.columns]
            display_cols = param_cols + metric_cols

            st.dataframe(
                top_df[display_cols].style.format({
                    col: "{:.4f}" for col in metric_cols
                }),
                use_container_width=True
            )

            # CPCV Validation Results
            # Handle different CPCV result formats
            cpcv_results_list = None
            if hasattr(result, 'top_k_cpcv_results') and result.top_k_cpcv_results:
                # FLAML format: list of CPCVResult objects
                cpcv_results_list = []
                top_k_results = result.get_top_k(top_k)
                for idx, (param_result, cpcv_result) in enumerate(zip(top_k_results, result.top_k_cpcv_results), 1):
                    cpcv_results_list.append({
                        'params': param_result.parameters,
                        'cpcv_result': cpcv_result
                    })
            elif hasattr(result, 'cpcv_results') and result.cpcv_results:
                # Grid search format: list of dicts
                cpcv_results_list = result.cpcv_results

            if cpcv_results_list:
                st.write("")
                st.caption("**CPCV Validation Results**")

                cpcv_data = []
                for idx, cpcv_item in enumerate(cpcv_results_list, 1):
                    cpcv_result = cpcv_item['cpcv_result']
                    cpcv_data.append({
                        'Rank': idx,
                        'Parameters': str(cpcv_item['params']),
                        'PBO': cpcv_result.overfitting_metrics.pbo,
                        'DSR': cpcv_result.overfitting_metrics.dsr,
                        'Degradation %': cpcv_result.overfitting_metrics.degradation_pct,
                        'Validation': '✅ Pass' if cpcv_result.passes_validation() else '❌ Fail'
                    })

                cpcv_df = pd.DataFrame(cpcv_data)
                st.dataframe(
                    cpcv_df.style.format({
                        'PBO': '{:.3f}',
                        'DSR': '{:.3f}',
                        'Degradation %': '{:.1f}%'
                    }).applymap(
                        lambda x: 'background-color: #d4edda' if x == '✅ Pass' else 'background-color: #f8d7da',
                        subset=['Validation']
                    ),
                    use_container_width=True
                )

                # Recommendation
                st.write("")
                st.caption("**Recommendation**")

                best_validated = None
                for cpcv_item in cpcv_results_list:
                    if cpcv_item['cpcv_result'].passes_validation():
                        best_validated = cpcv_item
                        break

                if best_validated:
                    st.success(f"✅ **Recommended parameters:** {best_validated['params']}")
                    st.write(f"   • PBO: {best_validated['cpcv_result'].overfitting_metrics.pbo:.3f} (< 0.50)")
                    st.write(f"   • DSR: {best_validated['cpcv_result'].overfitting_metrics.dsr:.3f} (> 1.0)")
                    st.write(f"   • Degradation: {best_validated['cpcv_result'].overfitting_metrics.degradation_pct:.1f}% (< 30%)")
                else:
                    st.warning("⚠️ None of the top parameters passed CPCV validation. Strategy may be overfit.")

        except Exception as e:
            st.error(f"Optimization failed: {str(e)}")
            import traceback
            st.code(traceback.format_exc())


def render_strategy_comparison_cpcv():
    """Compare multiple strategies with CPCV"""
    st.caption("**Strategy Comparison**")
    st.write("Compare different strategies using CPCV validation.")

    st.info("🚧 Feature coming soon! This will allow you to compare multiple strategies.")
    st.write("**Planned features:**")
    st.write("• Select multiple strategies to compare")
    st.write("• Run CPCV on each strategy")
    st.write("• Side-by-side comparison of PBO, DSR, Sharpe")
    st.write("• Identify which strategy is most robust")


# Helper function to get available strategies (already exists in main file)
# This is just a placeholder - use the existing get_available_strategies() function
def get_available_strategies():
    """Get available strategies from strategies module"""
    # This function already exists in the main streamlit_backtest_runner.py
    # We're just declaring it here for reference
    pass
