"""
Shared visualization functions for all CF algorithms
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl
from pathlib import Path

def _normalize_column_name(col: str) -> str:
    """Normalize column names to handle case variations"""
    return col.upper()

def _get_column(df: pl.DataFrame, col_name: str):
    """Get column data handling case variations"""
    normalized = _normalize_column_name(col_name)
    for actual_col in df.columns:
        if _normalize_column_name(actual_col) == normalized:
            return df[actual_col].to_list()
    raise KeyError(f"Column {col_name} not found. Available: {df.columns}")

def _find_column(df: pl.DataFrame, col_name: str) -> str:
    """Find actual column name handling case variations"""
    normalized = _normalize_column_name(col_name)
    for actual_col in df.columns:
        if _normalize_column_name(actual_col) == normalized:
            return actual_col
    raise KeyError(f"Column {col_name} not found. Available: {df.columns}")

def visualize_hyperparameter_tuning(df_results: pl.DataFrame, category: str,
                                    param_col: str, param_name: str,
                                    save_dir: Path, algo_name: str):
    """Comprehensive hyperparameter tuning visualization"""
    
    param_vals = df_results[param_col].to_list()
    ndcg20_col = _find_column(df_results, 'NDCG@20')
    best_val = df_results[param_col][df_results[ndcg20_col].arg_max()]
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    
    # Row 1: NDCG, Recall, MAP (each with @10, @20, @50)
    for row_idx, metric_base in enumerate(['NDCG', 'RECALL', 'MAP']):
        ax = axes[0, row_idx]
        colors = ['#1f77b4', '#2E86C1', '#5DADE2'] if row_idx == 0 else \
                 ['#27AE60', '#28B463', '#52BE80'] if row_idx == 1 else \
                 ['#8E44AD', '#9B59B6', '#BB8FCE']
        markers = ['o', 's', '^']
        
        for k_idx, (k, color, marker) in enumerate(zip([10, 20, 50], colors, markers)):
            col_name = f'{metric_base}@{k}'
            try:
                actual_col = _find_column(df_results, col_name)
                ax.plot(param_vals, df_results[actual_col].to_list(),
                       marker=marker, label=f'{metric_base}@{k}', linewidth=2.5,
                       markersize=8, color=color, alpha=0.8)
            except KeyError:
                pass
        
        ax.axvline(best_val, color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax.set_xlabel(param_name, fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{metric_base} Score', fontsize=12, fontweight='bold')
        ax.set_title(f'{metric_base}@K', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc='best')
    
    # Row 2: RMSE, Accuracy
    try:
        rmse_col = _find_column(df_results, 'RMSE')
        axes[1, 0].plot(param_vals, df_results[rmse_col].to_list(),
                       marker='d', linewidth=2.5, markersize=8, color='#E74C3C', alpha=0.8)
        axes[1, 0].axvline(best_val, color='red', linestyle='--', alpha=0.7, linewidth=2)
        axes[1, 0].set_xlabel(param_name, fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('RMSE', fontsize=12, fontweight='bold')
        axes[1, 0].set_title('RMSE (Lower is better)', fontsize=13, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
    except KeyError:
        pass
    
    try:
        acc_col = _find_column(df_results, 'Accuracy')
        axes[1, 1].plot(param_vals, df_results[acc_col].to_list(),
                       marker='p', linewidth=2.5, markersize=8, color='#F39C12', alpha=0.8)
        axes[1, 1].axvline(best_val, color='red', linestyle='--', alpha=0.7, linewidth=2)
        axes[1, 1].set_xlabel(param_name, fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        axes[1, 1].set_title('Accuracy', fontsize=13, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
    except KeyError:
        pass
    
    axes[1, 2].axis('off')
    
    # Row 3: All ranking metrics overlay (@20), Summary table
    metrics_to_plot = [
        ('NDCG@20', 'o', '#2E86C1', 'NDCG@20'),
        ('RECALL@20', 's', '#28B463', 'Recall@20'),
        ('MAP@20', '^', '#9B59B6', 'MAP@20')
    ]
    
    for metric, marker, color, label in metrics_to_plot:
        try:
            actual_col = _find_column(df_results, metric)
            axes[2, 0].plot(param_vals, df_results[actual_col].to_list(),
                           marker=marker, label=label, linewidth=2.5,
                           markersize=8, color=color, alpha=0.8)
        except KeyError:
            pass
    
    axes[2, 0].axvline(best_val, color='red', linestyle='--', alpha=0.7,
                      linewidth=2, label=f'Best={best_val}')
    axes[2, 0].set_xlabel(param_name, fontsize=12, fontweight='bold')
    axes[2, 0].set_ylabel('Score', fontsize=12, fontweight='bold')
    axes[2, 0].set_title('All Ranking Metrics @20', fontsize=13, fontweight='bold')
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].legend(fontsize=10, loc='best')
    
    # Summary table
    axes[2, 1].axis('off')
    
    try:
        ndcg20_col = _find_column(df_results, 'NDCG@20')
        recall20_col = _find_column(df_results, 'RECALL@20')
        map20_col = _find_column(df_results, 'MAP@20')
        rmse_col = _find_column(df_results, 'RMSE')
        
        table_cols = [param_col, ndcg20_col, recall20_col, map20_col, rmse_col]
        formatted_data = []
        for row in df_results.select(table_cols).iter_rows(named=True):
            formatted_data.append([
                row[param_col],
                f"{row[ndcg20_col]:.4f}",
                f"{row[recall20_col]:.4f}",
                f"{row[map20_col]:.4f}",
                f"{row[rmse_col]:.4f}"
            ])
        
        table = axes[2, 1].table(
            cellText=formatted_data,
            colLabels=[param_col, 'NDCG@20', 'Recall@20', 'MAP@20', 'RMSE'],
            cellLoc='center', loc='center',
            colWidths=[0.15, 0.2, 0.2, 0.2, 0.2]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2.5)
        
        best_idx = df_results[ndcg20_col].arg_max() + 1
        for j in range(5):
            table[(best_idx, j)].set_facecolor('#90EE90')
            table[(best_idx, j)].set_text_props(weight='bold')
    except KeyError as e:
        axes[2, 1].text(0.5, 0.5, f'Table error: {e}', ha='center', va='center')
    
    axes[2, 2].axis('off')
    
    plt.suptitle(f'Hyperparameter Tuning Results ({algo_name}) - {category}',
                fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    out_path = save_dir / f'{param_col.lower()}_tuning_{category}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return out_path

def visualize_final_results(results_list: list, save_dir: Path,
                           algo_name: str, k_values: list = [10, 20, 50]):
    """Visualize final evaluation results"""
    if not results_list:
        return
    
    df_results = pl.DataFrame(results_list)
    categories = df_results['category'].to_list()
    
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # RMSE and Accuracy
    ax1 = axes[0, 0]
    x = np.arange(len(categories))
    width = 0.35
    ax1.bar(x - width/2, df_results['rmse'].to_list(), width, label='RMSE', alpha=0.8)
    ax1_twin = ax1.twinx()
    ax1_twin.bar(x + width/2, df_results['accuracy'].to_list(), width,
                 label='Accuracy', alpha=0.8, color='orange')
    ax1.set_xlabel('Category')
    ax1.set_ylabel('RMSE', color='blue')
    ax1_twin.set_ylabel('Accuracy', color='orange')
    ax1.set_title(f'RMSE and Accuracy ({algo_name})')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, rotation=45, ha='right')
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    
    # Recall@K
    ax2 = axes[0, 1]
    for k in k_values:
        ax2.plot(categories, df_results[f'recall@{k}'].to_list(),
                marker='o', label=f'Recall@{k}')
    ax2.set_xlabel('Category')
    ax2.set_ylabel('Recall@K')
    ax2.set_title(f'Recall@K ({algo_name})')
    ax2.legend()
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # NDCG@K
    ax3 = axes[1, 0]
    for k in k_values:
        ax3.plot(categories, df_results[f'ndcg@{k}'].to_list(),
                marker='s', label=f'NDCG@{k}')
    ax3.set_xlabel('Category')
    ax3.set_ylabel('NDCG@K')
    ax3.set_title(f'NDCG@K ({algo_name})')
    ax3.legend()
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # MAP@K
    ax4 = axes[1, 1]
    for k in k_values:
        ax4.plot(categories, df_results[f'map@{k}'].to_list(),
                marker='^', label=f'MAP@{k}')
    ax4.set_xlabel('Category')
    ax4.set_ylabel('MAP@K')
    ax4.set_title(f'MAP@K ({algo_name})')
    ax4.legend()
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    out_plot = save_dir / 'evaluation_results.png'
    plt.savefig(out_plot, dpi=300, bbox_inches='tight')
    plt.show()
    
    return out_plot

def visualize_val_test_comparison(cat: str, param_val, tuning_row: dict,
                                  final_row: dict, save_dir: Path,
                                  param_name: str, algo_name: str):
    """Create validation vs test comparison"""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Get metrics with case-insensitive lookup
    def get_metric(row, metric_name):
        for key in row.keys():
            if key.upper() == metric_name.upper():
                return row[key]
        return 0.0
    
    metrics_names = ['NDCG@20', 'Recall@20', 'MAP@20']
    val_scores = [
        get_metric(tuning_row, 'NDCG@20'),
        get_metric(tuning_row, 'RECALL@20'),
        get_metric(tuning_row, 'MAP@20')
    ]
    test_scores = [
        get_metric(final_row, 'ndcg@20'),
        get_metric(final_row, 'recall@20'),
        get_metric(final_row, 'map@20')
    ]
    
    # Bar comparison
    x = np.arange(len(metrics_names))
    width = 0.35
    
    axes[0].bar(x - width/2, val_scores, width, label='Validation', alpha=0.8, color='#3498DB')
    axes[0].bar(x + width/2, test_scores, width, label='Test', alpha=0.8, color='#2ECC71')
    axes[0].set_xlabel('Metrics', fontsize=11)
    axes[0].set_ylabel('Score', fontsize=11)
    axes[0].set_title(f'Val vs Test ({algo_name}) - {cat} ({param_name}={param_val})',
                    fontsize=12, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics_names)
    axes[0].legend(fontsize=10)
    axes[0].grid(axis='y', alpha=0.3)
    
    for i, (v, t) in enumerate(zip(val_scores, test_scores)):
        axes[0].text(i - width/2, v, f'{v:.4f}', ha='center', va='bottom', fontsize=9)
        axes[0].text(i + width/2, t, f'{t:.4f}', ha='center', va='bottom', fontsize=9)
    
    # Table
    axes[1].axis('off')
    
    val_rmse = get_metric(tuning_row, 'RMSE')
    test_rmse = get_metric(final_row, 'rmse')
    
    improvements = [f"{(t/v - 1)*100:+.1f}%" if v > 0 else "N/A"
                   for v, t in zip(val_scores, test_scores)]
    
    val_scores_full = val_scores + [val_rmse]
    test_scores_full = test_scores + [test_rmse]
    improvements_full = improvements + [f"{(test_rmse/val_rmse - 1)*100:+.1f}%" if val_rmse > 0 else "N/A"]
    metrics_names_full = metrics_names + ['RMSE']
    
    table_data = [
        ['Metric', 'Validation', 'Test', 'Change'],
        *[[name, f"{v:.4f}", f"{t:.4f}", imp]
          for name, v, t, imp in zip(metrics_names_full, val_scores_full,
                                     test_scores_full, improvements_full)]
    ]
    
    table = axes[1].table(cellText=table_data, cellLoc='center', loc='center',
                         colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    for j in range(4):
        table[(0, j)].set_facecolor('#34495E')
        table[(0, j)].set_text_props(weight='bold', color='white')
    
    for i in range(1, 5):
        change_val = table_data[i][3]
        color = '#D5F4E6' if (change_val.startswith('+') and i < 4) or \
                            (change_val.startswith('-') and i == 4) else '#FADBD8'
        table[(i, 3)].set_facecolor(color)
    
    plt.tight_layout()
    out_path = save_dir / f'val_vs_test_{cat}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return out_path