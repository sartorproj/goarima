#!/usr/bin/env python3
"""
Visualize ARIMA/SARIMA forecast results using Plotly.
Generates interactive HTML dashboards and optional static PNG images.

Usage:
    python visualize.py           # Generate HTML only
    python visualize.py --png     # Generate HTML + PNG images
    python visualize.py --png-only # Generate PNG images only
"""

import argparse
import json
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COLORS = ['#ef4444', '#f97316', '#8b5cf6', '#06b6d4', '#84cc16', '#ec4899']

# Datasets to export as PNG (subset for README)
PNG_DATASETS = ['australian_population', 'australian_beer', 'us_house_sales', 'google_stock']


def load_data(json_file="forecast_results.json"):
    """Load forecast results from JSON file."""
    with open(json_file, 'r') as f:
        return json.load(f)


def create_forecast_chart(data, for_png=False):
    """Create forecast comparison chart."""
    fig = go.Figure()

    # Training data
    fig.add_trace(go.Scatter(
        x=data['train_index'],
        y=data['train_data'],
        mode='lines+markers',
        name='Training Data',
        line=dict(color='#3b82f6', width=2),
        marker=dict(size=4),
        hovertemplate='Period: %{x}<br>Value: %{y:.2f}<extra>Training</extra>' if not for_png else None
    ))

    # Test data (actual)
    fig.add_trace(go.Scatter(
        x=data['test_index'],
        y=data['test_data'],
        mode='lines+markers',
        name='Actual (Test)',
        line=dict(color='#22c55e', width=3),
        marker=dict(size=8, symbol='circle'),
        hovertemplate='Period: %{x}<br>Value: %{y:.2f}<extra>Actual</extra>' if not for_png else None
    ))

    # Model forecasts
    for i, model in enumerate(data['models']):
        color = COLORS[i % len(COLORS)]
        fig.add_trace(go.Scatter(
            x=data['test_index'],
            y=model['forecasts'],
            mode='lines+markers',
            name=f"{model['model_name']}{model['order']}",
            line=dict(color=color, width=2, dash='dash'),
            marker=dict(size=6, symbol='diamond'),
            hovertemplate=f"Period: %{{x}}<br>Forecast: %{{y:.2f}}<extra>{model['model_name']}</extra>" if not for_png else None
        ))

    # Train/test split line
    if data['train_index']:
        split_x = data['train_index'][-1]
        fig.add_vline(x=split_x, line_dash="dot", line_color="gray",
                     annotation_text="Train/Test Split", annotation_position="top")

    # Layout differs for PNG vs HTML
    if for_png:
        fig.update_layout(
            title=dict(text=f"{data['name']} - Forecast Comparison", font=dict(size=20)),
            xaxis_title='Time Period',
            yaxis_title='Value',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
            template='plotly_white',
            height=500, width=1000,
            margin=dict(t=100, b=60, l=80, r=40)
        )
    else:
        fig.update_layout(
            title='Forecast Comparison',
            xaxis_title='Time Period',
            yaxis_title='Value',
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
            template='plotly_white',
            height=450,
            margin=dict(t=80, b=50)
        )

    return fig


def create_metrics_table(data):
    """Create metrics comparison table for HTML."""
    models = data['models']
    has_suggested = any(m.get('suggested_order') for m in models)

    headers = ['Model', 'AIC', 'AICc', 'BIC', 'RMSE', 'MAE', 'MAPE']
    values = [
        [f"{m['model_name']}{m['order']}" for m in models],
        [f"{m['aic']:.2f}" for m in models],
        [f"{m.get('aicc', m['aic']):.2f}" for m in models],
        [f"{m['bic']:.2f}" for m in models],
        [f"{m['rmse']:.4f}" for m in models],
        [f"{m['mae']:.4f}" for m in models],
        [f"{m['mape']:.2f}%" for m in models],
    ]

    if has_suggested:
        headers.append('ACF/PACF Suggested')
        values.append([m.get('suggested_order', '-') or '-' for m in models])

    if any(m.get('models_evaluated') for m in models):
        headers.append('Models Evaluated')
        values.append([str(m.get('models_evaluated', '-')) if m.get('models_evaluated') else '-' for m in models])

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=[f'<b>{h}</b>' for h in headers],
            fill_color='#3b82f6',
            font=dict(color='white', size=13),
            align='center', height=35
        ),
        cells=dict(
            values=values,
            fill_color=[['#f8fafc', '#ffffff'] * ((len(models) + 1) // 2)][:len(models)],
            align='center', font=dict(size=12), height=30
        )
    )])

    fig.update_layout(
        title='Model Performance Metrics',
        height=150 + len(models) * 35,
        margin=dict(t=50, b=20, l=20, r=20)
    )
    return fig


def create_metrics_bar_chart(data):
    """Create metrics bar chart for PNG export."""
    models = data['models']
    if not models:
        return None

    model_names = [f"{m['model_name']}{m['order']}" for m in models]
    rmse_values = [m['rmse'] for m in models]
    mae_values = [m['mae'] for m in models]
    bar_colors = ['#3b82f6', '#22c55e', '#f97316', '#8b5cf6', '#06b6d4', '#84cc16'][:len(models)]

    fig = make_subplots(rows=1, cols=2, subplot_titles=['RMSE (lower is better)', 'MAE (lower is better)'])

    fig.add_trace(go.Bar(x=model_names, y=rmse_values, marker_color=bar_colors, showlegend=False), row=1, col=1)
    fig.add_trace(go.Bar(x=model_names, y=mae_values, marker_color=bar_colors, showlegend=False), row=1, col=2)

    fig.update_layout(
        title=dict(text=f"{data['name']} - Model Performance", font=dict(size=20)),
        template='plotly_white',
        height=400, width=1000,
        margin=dict(t=100, b=80, l=80, r=40)
    )
    return fig


def create_stationarity_table(data):
    """Create stationarity analysis table."""
    stationarity = data.get('stationarity', {})
    if not stationarity:
        return None

    rows = []
    if 'adf_statistic' in stationarity:
        pval = stationarity.get('adf_pvalue')
        rows.append(['ADF Test', f"{stationarity['adf_statistic']:.4f}",
                    f"{pval:.4f}" if isinstance(pval, (int, float)) else 'N/A',
                    '✓ Stationary' if stationarity.get('adf_stationary') else '✗ Non-stationary'])
    if 'kpss_statistic' in stationarity:
        pval = stationarity.get('kpss_pvalue')
        rows.append(['KPSS Test', f"{stationarity['kpss_statistic']:.4f}",
                    f"{pval:.4f}" if isinstance(pval, (int, float)) else 'N/A',
                    '✓ Stationary' if stationarity.get('kpss_stationary') else '✗ Non-stationary'])
    if 'ndiffs' in stationarity:
        rows.append(['ndiffs (recommended)', str(stationarity['ndiffs']), '-', f"d = {stationarity['ndiffs']}"])
    if 'nsdiffs' in stationarity:
        rows.append(['nsdiffs (recommended)', str(stationarity['nsdiffs']), '-', f"D = {stationarity['nsdiffs']}"])

    if not rows:
        return None

    values = list(zip(*rows))
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['<b>Test</b>', '<b>Statistic</b>', '<b>P-Value</b>', '<b>Result</b>'],
            fill_color='#10b981',
            font=dict(color='white', size=13),
            align='center', height=35
        ),
        cells=dict(
            values=values,
            fill_color=[['#f0fdf4', '#ffffff'] * ((len(rows) + 1) // 2)][:len(rows)],
            align='center', font=dict(size=12), height=30
        )
    )])

    fig.update_layout(title='Stationarity Analysis', height=150 + len(rows) * 35, margin=dict(t=50, b=20, l=20, r=20))
    return fig


def create_residual_chart(data):
    """Create residual analysis chart."""
    models = data['models']
    test_data = data['test_data']
    n_models = len(models)

    if n_models == 0:
        return None

    fig = make_subplots(
        rows=n_models, cols=2,
        subplot_titles=[item for m in models for item in
                       [f"{m['model_name']}{m['order']} - Residuals",
                        f"{m['model_name']}{m['order']} - Distribution"]],
        horizontal_spacing=0.1, vertical_spacing=0.15
    )

    for i, model in enumerate(models):
        row = i + 1
        color = COLORS[i % len(COLORS)]
        min_len = min(len(test_data), len(model['forecasts']))
        residuals = [test_data[j] - model['forecasts'][j] for j in range(min_len)]

        fig.add_trace(go.Scatter(
            x=list(range(1, len(residuals) + 1)), y=residuals,
            mode='lines+markers', line=dict(color=color), marker=dict(size=6),
            showlegend=False, hovertemplate='Period: %{x}<br>Residual: %{y:.4f}<extra></extra>'
        ), row=row, col=1)

        fig.add_shape(type="line", x0=0, x1=len(residuals) + 1, y0=0, y1=0,
                     line=dict(color="gray", width=1, dash="dash"), row=row, col=1)

        fig.add_trace(go.Histogram(x=residuals, marker_color=color, opacity=0.7, showlegend=False), row=row, col=2)

    fig.update_layout(title='Residual Analysis', height=250 * n_models, template='plotly_white', margin=dict(t=80, b=50))
    return fig


def create_dashboard_html(data, output_file):
    """Create HTML dashboard with all charts."""
    dataset_name = data.get('name', 'Time Series Analysis')
    description = data.get('description', '')

    forecast_fig = create_forecast_chart(data)
    metrics_fig = create_metrics_table(data)
    stationarity_fig = create_stationarity_table(data)
    residual_fig = create_residual_chart(data)

    forecast_html = forecast_fig.to_html(full_html=False, include_plotlyjs=False)
    metrics_html = metrics_fig.to_html(full_html=False, include_plotlyjs=False)
    stationarity_html = stationarity_fig.to_html(full_html=False, include_plotlyjs=False) if stationarity_fig else ""
    residual_html = residual_fig.to_html(full_html=False, include_plotlyjs=False) if residual_fig else ""

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{dataset_name} - GoARIMA Analysis</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background-color: #f8fafc; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        .header {{ text-align: center; padding: 20px; background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%); color: white; border-radius: 12px; margin-bottom: 20px; }}
        .header h1 {{ margin: 0 0 10px 0; font-size: 28px; }}
        .header p {{ margin: 0; opacity: 0.9; font-size: 16px; }}
        .chart-section {{ background: white; border-radius: 12px; padding: 20px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .tables-row {{ display: flex; gap: 20px; flex-wrap: wrap; }}
        .tables-row > div {{ flex: 1; min-width: 400px; }}
        .footer {{ text-align: center; padding: 20px; color: #64748b; font-size: 14px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header"><h1>{dataset_name}</h1><p>{description}</p></div>
        <div class="chart-section">{forecast_html}</div>
        <div class="tables-row">
            <div class="chart-section">{metrics_html}</div>
            <div class="chart-section">{stationarity_html if stationarity_html else "<p style='text-align:center;color:#94a3b8;'>No stationarity analysis available</p>"}</div>
        </div>
        <div class="chart-section">{residual_html if residual_html else "<p style='text-align:center;color:#94a3b8;'>No residual analysis available</p>"}</div>
        <div class="footer">Generated by GoARIMA | <a href="https://otexts.com/fpppy/nbs/09-arima.html">Reference: Forecasting: Principles and Practice</a></div>
    </div>
</body>
</html>"""

    with open(output_file, 'w') as f:
        f.write(html_content)


def export_png(datasets, output_dir="images"):
    """Export PNG images for selected datasets."""
    os.makedirs(output_dir, exist_ok=True)
    exported = []

    for dataset in datasets:
        safe_name = dataset.get('name', 'Unknown').lower().replace(' ', '_').replace('(', '').replace(')', '')

        # Only export selected datasets (or all if <= 4)
        if len(datasets) > 4 and safe_name not in PNG_DATASETS:
            continue

        print(f"\n  Exporting PNG: {dataset['name']}")

        # Forecast chart
        fig = create_forecast_chart(dataset, for_png=True)
        forecast_file = f"{output_dir}/{safe_name}_forecast.png"
        fig.write_image(forecast_file, scale=2)
        print(f"    ✓ {forecast_file}")
        exported.append(forecast_file)

        # Metrics bar chart
        metrics_fig = create_metrics_bar_chart(dataset)
        if metrics_fig:
            metrics_file = f"{output_dir}/{safe_name}_metrics.png"
            metrics_fig.write_image(metrics_file, scale=2)
            print(f"    ✓ {metrics_file}")
            exported.append(metrics_file)

    return exported


def main():
    parser = argparse.ArgumentParser(description='Visualize ARIMA forecast results')
    parser.add_argument('--png', action='store_true', help='Also export PNG images')
    parser.add_argument('--png-only', action='store_true', help='Only export PNG images (skip HTML)')
    args = parser.parse_args()

    print("=" * 60)
    print("GoARIMA Visualization")
    print("=" * 60)

    json_file = "forecast_results.json"
    if not os.path.exists(json_file):
        print(f"\nError: {json_file} not found!")
        print("Run the Go demo first: go run .")
        return

    print(f"\nLoading data from {json_file}...")
    raw_data = load_data(json_file)
    datasets = raw_data.get('datasets', [raw_data])
    print(f"Found {len(datasets)} dataset(s)")

    # Generate HTML dashboards
    if not args.png_only:
        output_dir = "charts"
        os.makedirs(output_dir, exist_ok=True)

        # Clear old files
        for f in os.listdir(output_dir):
            if f.endswith('.html'):
                os.remove(os.path.join(output_dir, f))

        print("\nGenerating HTML dashboards...")
        for i, dataset in enumerate(datasets):
            safe_name = dataset.get('name', f'Dataset_{i+1}').lower().replace(' ', '_').replace('(', '').replace(')', '')
            filename = f"{output_dir}/{i+1}_{safe_name}.html"
            create_dashboard_html(dataset, filename)
            print(f"  ✓ {filename}")

        print(f"\nOpen in browser: open {output_dir}/1_*.html")

    # Generate PNG images
    if args.png or args.png_only:
        print("\nGenerating PNG images...")
        exported = export_png(datasets)
        print(f"\nExported {len(exported)} PNG files to images/")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
