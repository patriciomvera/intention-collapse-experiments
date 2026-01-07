"""
Visualization Utilities for Intention Collapse Experiments

This module creates publication-quality figures for presenting
experimental results. All figures are designed to be suitable
for inclusion in academic papers.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import warnings

# Suppress matplotlib warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Publication-quality settings
FIGURE_PARAMS = {
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 13,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
}

# Color scheme
COLORS = {
    'baseline': '#1f77b4',  # Blue
    'enhanced': '#ff7f0e',  # Orange
    'correct': '#2ca02c',   # Green
    'incorrect': '#d62728', # Red
    'neutral': '#7f7f7f',   # Gray
}


def setup_style():
    """Set up matplotlib style for publication figures."""
    plt.rcParams.update(FIGURE_PARAMS)
    sns.set_style("whitegrid")


def save_figure(
    fig: plt.Figure,
    filename: str,
    output_dir: str = "results/figures",
    formats: List[str] = ["pdf", "png"],
    dpi: int = 300
):
    """
    Save figure in multiple formats.
    
    Args:
        fig: Matplotlib figure
        filename: Base filename (without extension)
        output_dir: Output directory
        formats: List of formats to save
        dpi: Resolution for raster formats
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for fmt in formats:
        filepath = output_path / f"{filename}.{fmt}"
        fig.savefig(
            filepath,
            format=fmt,
            dpi=dpi,
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none'
        )
        print(f"Saved: {filepath}")


def plot_metrics_comparison(
    baseline_metrics: Dict[str, List[float]],
    enhanced_metrics: Dict[str, List[float]],
    metric_names: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (10, 4)
) -> plt.Figure:
    """
    Create comparison plot of intention metrics between conditions.
    
    Args:
        baseline_metrics: Dict mapping metric names to lists of values
        enhanced_metrics: Dict mapping metric names to lists of values
        metric_names: Order of metrics to plot (None for all)
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    setup_style()
    
    if metric_names is None:
        metric_names = list(baseline_metrics.keys())
    
    n_metrics = len(metric_names)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    
    if n_metrics == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, metric_names):
        baseline_vals = baseline_metrics[metric]
        enhanced_vals = enhanced_metrics[metric]
        
        # Create box plot
        data = [baseline_vals, enhanced_vals]
        positions = [0, 1]
        
        bp = ax.boxplot(
            data,
            positions=positions,
            widths=0.6,
            patch_artist=True
        )
        
        # Color boxes
        bp['boxes'][0].set_facecolor(COLORS['baseline'])
        bp['boxes'][1].set_facecolor(COLORS['enhanced'])
        for box in bp['boxes']:
            box.set_alpha(0.7)
        
        # Add individual points with jitter
        for i, (vals, pos) in enumerate(zip(data, positions)):
            jitter = np.random.normal(0, 0.04, len(vals))
            color = COLORS['baseline'] if i == 0 else COLORS['enhanced']
            ax.scatter(
                pos + jitter, vals,
                alpha=0.3, s=20, color=color
            )
        
        ax.set_xticks(positions)
        ax.set_xticklabels(['Baseline', 'CoT'])
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(metric.replace('_', ' ').title())
        
        # Add mean values as text
        for i, vals in enumerate(data):
            mean_val = np.mean(vals)
            ax.text(
                positions[i], ax.get_ylim()[1] * 0.95,
                f'μ={mean_val:.2f}',
                ha='center', va='top', fontsize=8
            )
    
    plt.tight_layout()
    return fig


def plot_correlation_matrix(
    metrics_data: Dict[str, List[float]],
    correctness: List[bool],
    figsize: Tuple[float, float] = (8, 6)
) -> plt.Figure:
    """
    Create correlation matrix heatmap between metrics and correctness.
    
    Args:
        metrics_data: Dict mapping metric names to values
        correctness: List of boolean correctness labels
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    setup_style()
    
    # Combine all data
    data = {**metrics_data, 'correct': [int(c) for c in correctness]}
    
    # Create DataFrame-like structure for correlation
    keys = list(data.keys())
    n = len(keys)
    
    corr_matrix = np.zeros((n, n))
    for i, key_i in enumerate(keys):
        for j, key_j in enumerate(keys):
            vals_i = np.array(data[key_i])
            vals_j = np.array(data[key_j])
            corr_matrix[i, j] = np.corrcoef(vals_i, vals_j)[0, 1]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Correlation')
    
    # Set ticks
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    labels = [k.replace('_', '\n') for k in keys]
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    
    # Add correlation values as text
    for i in range(n):
        for j in range(n):
            val = corr_matrix[i, j]
            color = 'white' if abs(val) > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                   color=color, fontsize=9)
    
    ax.set_title('Correlation Matrix: Metrics vs. Correctness')
    plt.tight_layout()
    
    return fig


def plot_entropy_trajectory(
    trajectories: List[List[float]],
    correct_mask: Optional[List[bool]] = None,
    figsize: Tuple[float, float] = (8, 5)
) -> plt.Figure:
    """
    Plot entropy trajectories during generation.
    
    Shows the U-shaped pattern predicted by the framework:
    entropy rises during exploration, then falls as intention crystallizes.
    
    Args:
        trajectories: List of entropy trajectories for each example
        correct_mask: Optional mask to color by correctness
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    setup_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Normalize trajectory lengths by interpolation
    max_len = max(len(t) for t in trajectories)
    normalized = []
    
    for traj in trajectories:
        if len(traj) < 2:
            continue
        x_old = np.linspace(0, 1, len(traj))
        x_new = np.linspace(0, 1, max_len)
        normalized.append(np.interp(x_new, x_old, traj))
    
    normalized = np.array(normalized)
    
    # Plot individual trajectories
    for i, traj in enumerate(normalized):
        if correct_mask is not None:
            color = COLORS['correct'] if correct_mask[i] else COLORS['incorrect']
            alpha = 0.2
        else:
            color = COLORS['neutral']
            alpha = 0.3
        ax.plot(traj, color=color, alpha=alpha, linewidth=0.5)
    
    # Plot mean trajectory
    mean_traj = np.mean(normalized, axis=0)
    ax.plot(mean_traj, color='black', linewidth=2, label='Mean')
    
    # Add standard deviation band
    std_traj = np.std(normalized, axis=0)
    x = np.arange(len(mean_traj))
    ax.fill_between(x, mean_traj - std_traj, mean_traj + std_traj,
                    alpha=0.2, color='black')
    
    ax.set_xlabel('Generation Progress (normalized)')
    ax.set_ylabel('Intention Entropy (bits)')
    ax.set_title('Entropy Trajectory During Generation')
    
    if correct_mask is not None:
        # Add legend for correct/incorrect
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=COLORS['correct'], alpha=0.5, label='Correct'),
            Patch(facecolor=COLORS['incorrect'], alpha=0.5, label='Incorrect')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    return fig


def plot_dim_eff_vs_accuracy(
    dim_effs: List[int],
    accuracies: List[bool],
    condition_labels: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (7, 5)
) -> plt.Figure:
    """
    Scatter plot of effective dimensionality vs accuracy.
    
    Args:
        dim_effs: Effective dimensionality values
        accuracies: Boolean correctness labels
        condition_labels: Optional condition labels for coloring
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    setup_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    dim_effs = np.array(dim_effs)
    accuracies = np.array(accuracies).astype(int)
    
    # Bin dim_eff and compute accuracy per bin
    bins = np.percentile(dim_effs, np.linspace(0, 100, 11))
    bins = np.unique(bins)
    
    bin_centers = []
    bin_accuracies = []
    bin_counts = []
    
    for i in range(len(bins) - 1):
        mask = (dim_effs >= bins[i]) & (dim_effs < bins[i+1])
        if mask.sum() > 0:
            bin_centers.append((bins[i] + bins[i+1]) / 2)
            bin_accuracies.append(accuracies[mask].mean())
            bin_counts.append(mask.sum())
    
    # Scatter with size by count
    sizes = np.array(bin_counts) * 10
    ax.scatter(bin_centers, bin_accuracies, s=sizes, alpha=0.7,
              color=COLORS['enhanced'])
    
    # Fit trend line
    if len(bin_centers) > 2:
        z = np.polyfit(bin_centers, bin_accuracies, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(bin_centers), max(bin_centers), 100)
        ax.plot(x_line, p(x_line), '--', color=COLORS['baseline'],
               label=f'Trend (slope={z[0]:.3f})')
        ax.legend()
    
    ax.set_xlabel('Effective Dimensionality (dim_eff)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Relationship Between Intention Richness and Performance')
    
    plt.tight_layout()
    return fig


def plot_recoverability_comparison(
    pre_collapse_acc: float,
    post_collapse_acc: float,
    pre_collapse_std: Optional[float] = None,
    figsize: Tuple[float, float] = (6, 5)
) -> plt.Figure:
    """
    Bar plot comparing recoverability before and after collapse.
    
    Args:
        pre_collapse_acc: Probe accuracy on pre-collapse activations
        post_collapse_acc: Accuracy of verbalized answers
        pre_collapse_std: Standard deviation of pre-collapse accuracy
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    setup_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    conditions = ['Pre-Collapse\n(Probe on I)', 'Post-Collapse\n(Verbalized)']
    accuracies = [pre_collapse_acc, post_collapse_acc]
    colors = [COLORS['enhanced'], COLORS['baseline']]
    
    bars = ax.bar(conditions, accuracies, color=colors, alpha=0.8, width=0.6)
    
    # Add error bar if provided
    if pre_collapse_std is not None:
        ax.errorbar(0, pre_collapse_acc, yerr=pre_collapse_std,
                   fmt='none', color='black', capsize=5)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{acc:.1%}', ha='center', va='bottom', fontsize=11)
    
    # Add gap annotation
    gap = pre_collapse_acc - post_collapse_acc
    mid_y = (pre_collapse_acc + post_collapse_acc) / 2
    ax.annotate(
        f'Gap: {gap:.1%}',
        xy=(0.5, mid_y), xytext=(1.3, mid_y),
        arrowprops=dict(arrowstyle='->', color='gray'),
        fontsize=10, va='center'
    )
    
    ax.set_ylabel('Accuracy / Recoverability')
    ax.set_title('Information Loss During Intention Collapse')
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    return fig


def create_publication_figure(
    results: Dict[str, Any],
    figure_type: str = "main",
    figsize: Tuple[float, float] = (12, 8)
) -> plt.Figure:
    """
    Create the main publication figure combining multiple plots.
    
    Args:
        results: Dictionary containing all experiment results
        figure_type: Type of figure ("main" or "supplementary")
        figsize: Figure size
        
    Returns:
        Combined matplotlib figure
    """
    setup_style()
    
    fig = plt.figure(figsize=figsize)
    
    if figure_type == "main":
        # 2x2 grid of main results
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # (A) Metrics comparison
        ax1 = fig.add_subplot(gs[0, 0])
        # Plot code here...
        ax1.set_title('(A) Intention Metrics by Condition')
        
        # (B) Correlation matrix
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_title('(B) Metric Correlations')
        
        # (C) Entropy trajectory
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.set_title('(C) Entropy Dynamics')
        
        # (D) Dim_eff vs accuracy
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.set_title('(D) Dimensionality vs Performance')
    
    plt.suptitle('Intention Collapse: Experimental Results', fontsize=14)
    
    return fig


def plot_layer_wise_probing(
    layer_results: Dict[str, float],
    layer_names: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (10, 4)
) -> plt.Figure:
    """
    Plot probe accuracy across different layers.
    
    Args:
        layer_results: Dict mapping layer names to probe accuracies
        layer_names: Order of layers to plot
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    setup_style()
    
    if layer_names is None:
        layer_names = list(layer_results.keys())
    
    accuracies = [layer_results[name] for name in layer_names]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(layer_names))
    bars = ax.bar(x, accuracies, color=COLORS['enhanced'], alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, rotation=45, ha='right')
    ax.set_ylabel('Probe Accuracy')
    ax.set_xlabel('Layer')
    ax.set_title('Layer-wise Information Content')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
    ax.legend()
    
    plt.tight_layout()
    return fig
