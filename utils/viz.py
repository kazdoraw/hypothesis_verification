"""
Visualization utilities for DS experiments.
"""

from pathlib import Path
from typing import Optional, Any
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    plt = None
    sns = None
    HAS_MATPLOTLIB = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


def _ensure_dir(path: Path) -> None:
    """Ensure parent directory exists."""
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list,
    save_path: Optional[str | Path] = None,
    title: str = "Confusion Matrix",
    figsize: tuple = (10, 8),
    cmap: str = "Blues",
    show: bool = True
) -> Optional[Any]:
    """
    Plot confusion matrix heatmap.
    
    Args:
        cm: Confusion matrix array
        labels: Class labels
        save_path: Path to save figure
        title: Plot title
        figsize: Figure size
        cmap: Colormap
        show: Whether to display the plot
        
    Returns:
        Matplotlib figure or None
    """
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not installed, skipping plot")
        return None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap=cmap,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax
    )
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    
    # Rotate labels
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        _ensure_dir(save_path)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_f1_by_class(
    scores_a: dict,
    scores_b: dict,
    labels: Optional[list] = None,
    save_path: Optional[str | Path] = None,
    title: str = "F1 Score by Class: Baseline A vs Proposed B",
    figsize: tuple = (12, 6),
    show: bool = True
) -> Optional[Any]:
    """
    Plot F1 scores comparison by class.
    
    Args:
        scores_a: Per-class F1 scores for baseline A
        scores_b: Per-class F1 scores for proposed B
        labels: Class labels (inferred from scores if None)
        save_path: Path to save figure
        title: Plot title
        figsize: Figure size
        show: Whether to display
        
    Returns:
        Matplotlib figure or None
    """
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not installed, skipping plot")
        return None
    
    if labels is None:
        labels = list(scores_a.keys())
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=figsize)
    
    vals_a = [scores_a.get(l, 0) for l in labels]
    vals_b = [scores_b.get(l, 0) for l in labels]
    
    bars_a = ax.bar(x - width/2, vals_a, width, label='Baseline A (LLM)', color='#ff7f0e', alpha=0.8)
    bars_b = ax.bar(x + width/2, vals_b, width, label='Proposed B (Rule+ML)', color='#1f77b4', alpha=0.8)
    
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)
    
    # Add value labels
    for bar in bars_a:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    for bar in bars_b:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    # Add horizontal line at 0.85 threshold
    ax.axhline(y=0.85, color='green', linestyle='--', alpha=0.7, label='Target (0.85)')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        _ensure_dir(save_path)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_completion_distribution(
    completion_a: list,
    completion_b: list,
    save_path: Optional[str | Path] = None,
    title: str = "Completion Rate Distribution: A vs B",
    figsize: tuple = (10, 6),
    show: bool = True
) -> Optional[Any]:
    """
    Plot completion rate distribution comparison.
    
    Args:
        completion_a: Completion rates for baseline A
        completion_b: Completion rates for proposed B
        save_path: Path to save figure
        title: Plot title
        figsize: Figure size
        show: Whether to display
        
    Returns:
        Matplotlib figure or None
    """
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not installed, skipping plot")
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Boxplot
    ax1 = axes[0]
    bp = ax1.boxplot(
        [completion_a, completion_b],
        labels=['Baseline A', 'Proposed B'],
        patch_artist=True
    )
    bp['boxes'][0].set_facecolor('#ff7f0e')
    bp['boxes'][1].set_facecolor('#1f77b4')
    ax1.set_ylabel('Completion Rate')
    ax1.set_title('Boxplot')
    ax1.axhline(y=0.9, color='green', linestyle='--', alpha=0.7, label='Target (0.9)')
    ax1.legend()
    
    # Histogram
    ax2 = axes[1]
    ax2.hist(completion_a, bins=10, alpha=0.7, label='Baseline A', color='#ff7f0e')
    ax2.hist(completion_b, bins=10, alpha=0.7, label='Proposed B', color='#1f77b4')
    ax2.axvline(x=0.9, color='green', linestyle='--', alpha=0.7, label='Target (0.9)')
    ax2.set_xlabel('Completion Rate')
    ax2.set_ylabel('Count')
    ax2.set_title('Histogram')
    ax2.legend()
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        _ensure_dir(save_path)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_economics_comparison(
    tokens_a: int,
    tokens_b: int,
    calls_a: int,
    calls_b: int,
    save_path: Optional[str | Path] = None,
    title: str = "Economics: LLM Calls and Tokens",
    figsize: tuple = (12, 5),
    show: bool = True
) -> Optional[Any]:
    """
    Plot economic comparison (tokens and LLM calls).
    
    Args:
        tokens_a, tokens_b: Total tokens for A and B
        calls_a, calls_b: Total LLM calls for A and B
        save_path: Path to save figure
        title: Plot title
        figsize: Figure size
        show: Whether to display
        
    Returns:
        Matplotlib figure or None
    """
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not installed, skipping plot")
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Tokens comparison
    ax1 = axes[0]
    bars1 = ax1.bar(['Baseline A', 'Proposed B'], [tokens_a, tokens_b], 
                    color=['#ff7f0e', '#1f77b4'], alpha=0.8)
    ax1.set_ylabel('Total Tokens')
    ax1.set_title('Token Usage')
    
    # Add reduction annotation
    if tokens_a > 0:
        reduction = (tokens_a - tokens_b) / tokens_a * 100
        ax1.annotate(f'-{reduction:.1f}%', 
                     xy=(1, tokens_b), 
                     xytext=(1.2, (tokens_a + tokens_b) / 2),
                     fontsize=12, color='green',
                     arrowprops=dict(arrowstyle='->', color='green'))
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{int(height):,}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom')
    
    # Calls comparison
    ax2 = axes[1]
    bars2 = ax2.bar(['Baseline A', 'Proposed B'], [calls_a, calls_b],
                    color=['#ff7f0e', '#1f77b4'], alpha=0.8)
    ax2.set_ylabel('Total LLM Calls')
    ax2.set_title('LLM Calls')
    
    if calls_a > 0:
        reduction = (calls_a - calls_b) / calls_a * 100
        ax2.annotate(f'-{reduction:.1f}%',
                     xy=(1, calls_b),
                     xytext=(1.2, (calls_a + calls_b) / 2),
                     fontsize=12, color='green',
                     arrowprops=dict(arrowstyle='->', color='green'))
    
    for bar in bars2:
        height = bar.get_height()
        ax2.annotate(f'{int(height):,}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom')
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        _ensure_dir(save_path)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_model_comparison(
    results: list[dict],
    metric: str = "macro_f1",
    save_path: Optional[str | Path] = None,
    title: str = "ML Model Comparison",
    figsize: tuple = (10, 6),
    show: bool = True
) -> Optional[Any]:
    """
    Plot comparison of multiple ML models.
    
    Args:
        results: List of dicts with 'model_name' and metric values
        metric: Metric to compare
        save_path: Path to save figure
        title: Plot title
        figsize: Figure size
        show: Whether to display
        
    Returns:
        Matplotlib figure or None
    """
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not installed, skipping plot")
        return None
    
    models = [r.get('model_name', f'Model {i}') for i, r in enumerate(results)]
    values = [r.get(metric, 0) for r in results]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(models)))
    bars = ax.barh(models, values, color=colors, alpha=0.8)
    
    ax.set_xlabel(metric.replace('_', ' ').title())
    ax.set_title(title)
    ax.set_xlim(0, 1.1)
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax.annotate(f'{val:.3f}',
                    xy=(val, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0),
                    textcoords="offset points",
                    ha='left', va='center')
    
    # Add threshold line
    ax.axvline(x=0.85, color='green', linestyle='--', alpha=0.7, label='Target (0.85)')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        _ensure_dir(save_path)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_turns_vs_completion(
    turns_a: list,
    completion_a: list,
    turns_b: list,
    completion_b: list,
    save_path: Optional[str | Path] = None,
    title: str = "Turns vs Completion Rate",
    figsize: tuple = (10, 6),
    show: bool = True
) -> Optional[Any]:
    """
    Plot scatter of turns vs completion rate.
    
    Args:
        turns_a, turns_b: Number of turns for each case
        completion_a, completion_b: Completion rates
        save_path: Path to save figure
        title: Plot title
        figsize: Figure size
        show: Whether to display
        
    Returns:
        Matplotlib figure or None
    """
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not installed, skipping plot")
        return None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.scatter(turns_a, completion_a, alpha=0.7, label='Baseline A', 
               color='#ff7f0e', s=100, marker='o')
    ax.scatter(turns_b, completion_b, alpha=0.7, label='Proposed B',
               color='#1f77b4', s=100, marker='^')
    
    ax.set_xlabel('Number of Turns')
    ax.set_ylabel('Completion Rate')
    ax.set_title(title)
    ax.legend()
    
    # Add target line
    ax.axhline(y=0.9, color='green', linestyle='--', alpha=0.7, label='Target (0.9)')
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        _ensure_dir(save_path)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    
    return fig


if __name__ == "__main__":
    # Test visualizations
    print("Testing visualizations...")
    
    # Test confusion matrix
    cm = np.array([[10, 2, 1], [1, 15, 0], [2, 1, 8]])
    labels = ['Class A', 'Class B', 'Class C']
    plot_confusion_matrix(cm, labels, title="Test Confusion Matrix", show=True)
    
    # Test F1 comparison
    scores_a = {'booking': 0.75, 'complaint': 0.70, 'other': 0.65}
    scores_b = {'booking': 0.90, 'complaint': 0.85, 'other': 0.80}
    plot_f1_by_class(scores_a, scores_b, show=True)
