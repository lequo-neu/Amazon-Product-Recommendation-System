"""
Shared visualization functions for all CF algorithms
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl
from pathlib import Path

# ============================================================================
# Shared Helper Functions
# ============================================================================

def create_metric_plot(ax, x_vals, y_vals, best_val, title, xlabel, ylabel, 
                      color, marker='o'):
    """Create a single metric plot (reusable for all algorithms)"""
    ax.plot(x_vals, y_vals, marker=marker, linewidth=2, color=color, markersize=8)
    ax.axvline(best_val, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)


def create_summary_table(ax, df_results, param_col, best_val):
    """Create summary table (reusable for all algorithms)"""
    ax.axis('off')
    
    # Select columns (handle both 'K' and 'n_factors')
    table_data = df_results.select([param_col, 'NDCG@20', 'Recall@20', 'MAP@20', 'RMSE'])
    
    formatted_data = [[row[param_col], f"{row['NDCG@20']:.4f}", 
                      f"{row['Recall@20']:.4f}", f"{row['MAP@20']:.4f}", 
                      f"{row['RMSE']:.4f}"]
                     for row in table_data.iter_rows(named=True)]
    
    table = ax.table(
        cellText=formatted_data,
        colLabels=[param_col, 'NDCG@20', 'Recall@20', 'MAP@20', 'RMSE'],
        cellLoc='center', loc='center', 
        colWidths=[0.15, 0.2, 0.2, 0.2, 0.2]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)
    
    # Highlight best row
    best_idx = df_results['NDCG@20'].arg_max() + 1
    for j in range(5):
        table[(best_idx, j)].set_facecolor('#90EE90')
        table[(best_idx, j)].set_text_props(weight='bold')


# def visualize_hyperparameter_tuning(df_results: pl.DataFrame, category: str,
#                                     param_col: str, param_name: str,
#                                     save_dir: Path, algo_name: str):
#     """
#     Comprehensive hyperparameter tuning visualization (works for all algorithms)
    
#     Args:
#         df_results: Tuning results DataFrame
#         category: Category name
#         param_col: Column name ('K' or 'n_factors')
#         param_name: Display name ('K (Neighbors)' or 'n_factors')
#         save_dir: Directory to save plot
#         algo_name: Algorithm name for filename
#     """
#     param_vals = df_results[param_col].to_list()
#     best_val = df_results[param_col][df_results['NDCG@20'].arg_max()]
    
#     fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
#     # Plot metrics
#     create_metric_plot(axes[0, 0], param_vals, df_results['NDCG@20'].to_list(),
#                       best_val, 'NDCG@20 (PRIMARY)', param_name, 'NDCG@20', '#2E86C1')
#     create_metric_plot(axes[0, 1], param_vals, df_results['Recall@20'].to_list(),
#                       best_val, 'Recall@20 (SECONDARY)', param_name, 'Recall@20', '#28B463', 's')
#     create_metric_plot(axes[0, 2], param_vals, df_results['MAP@20'].to_list(),
#                       best_val, 'MAP@20 (TERTIARY)', param_name, 'MAP@20', '#9B59B6', '^')
#     create_metric_plot(axes[1, 0], param_vals, df_results['RMSE'].to_list(),
#                       best_val, 'RMSE (Lower is better)', param_name, 'RMSE', '#E74C3C', 'd')
    
#     # All metrics overlay
#     for metric, marker, label in [('NDCG@20', 'o', 'NDCG@20'),
#                                    ('Recall@20', 's', 'Recall@20'),
#                                    ('MAP@20', '^', 'MAP@20')]:
#         axes[1, 1].plot(param_vals, df_results[metric].to_list(),
#                        marker=marker, label=label, linewidth=2, markersize=7)
#     axes[1, 1].axvline(best_val, color='red', linestyle='--', alpha=0.7,
#                       linewidth=2, label=f'Best={best_val}')
#     axes[1, 1].set_xlabel(param_name, fontsize=11)
#     axes[1, 1].set_ylabel('Score', fontsize=11)
#     axes[1, 1].set_title('All Ranking Metrics', fontsize=12, fontweight='bold')
#     axes[1, 1].grid(True, alpha=0.3)
#     axes[1, 1].legend(fontsize=9)
    
#     # Summary table
#     create_summary_table(axes[1, 2], df_results, param_col, best_val)
    
#     plt.suptitle(f'Hyperparameter Tuning Results ({algo_name}) - {category}',
#                 fontsize=16, fontweight='bold')
#     plt.tight_layout()
    
#     # Save
#     out_path = save_dir / f'{param_col.lower()}_tuning_{category}.png'
#     plt.savefig(out_path, dpi=150, bbox_inches='tight')
#     plt.show()
    
#     return out_path
def visualize_hyperparameter_tuning(df_results: pl.DataFrame, category: str,
                                    param_col: str, param_name: str,
                                    save_dir: Path, algo_name: str):
    """
    Comprehensive hyperparameter tuning visualization (works for all algorithms)
    
    Args:
        df_results: Tuning results DataFrame
        category: Category name
        param_col: Column name ('K' or 'n_factors')
        param_name: Display name ('K (Neighbors)' or 'n_factors')
        save_dir: Directory to save plot
        algo_name: Algorithm name for filename
    """
    param_vals = df_results[param_col].to_list()
    best_val = df_results[param_col][df_results['NDCG@20'].arg_max()]
    
    # Create figure with 3 rows x 3 columns
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    
    # ========== ROW 1: NDCG, Recall, MAP (each with @10, @20, @50) ==========
    
    # NDCG subplot
    for k, color, marker in [(10, '#1f77b4', 'o'), (20, '#2E86C1', 's'), (50, '#5DADE2', '^')]:
        col_name = f'NDCG@{k}'
        if col_name in df_results.columns:
            axes[0, 0].plot(param_vals, df_results[col_name].to_list(),
                           marker=marker, label=f'NDCG@{k}', linewidth=2.5, 
                           markersize=8, color=color, alpha=0.8)
    axes[0, 0].axvline(best_val, color='red', linestyle='--', alpha=0.7, linewidth=2)
    axes[0, 0].set_xlabel(param_name, fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('NDCG Score', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('NDCG@K', fontsize=13, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=10, loc='best')
    
    # Recall subplot
    for k, color, marker in [(10, '#27AE60', 'o'), (20, '#28B463', 's'), (50, '#52BE80', '^')]:
        col_name = f'Recall@{k}'
        if col_name in df_results.columns:
            axes[0, 1].plot(param_vals, df_results[col_name].to_list(),
                           marker=marker, label=f'Recall@{k}', linewidth=2.5,
                           markersize=8, color=color, alpha=0.8)
    axes[0, 1].axvline(best_val, color='red', linestyle='--', alpha=0.7, linewidth=2)
    axes[0, 1].set_xlabel(param_name, fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Recall Score', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Recall@K', fontsize=13, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=10, loc='best')
    
    # MAP subplot
    for k, color, marker in [(10, '#8E44AD', 'o'), (20, '#9B59B6', 's'), (50, '#BB8FCE', '^')]:
        col_name = f'MAP@{k}'
        if col_name in df_results.columns:
            axes[0, 2].plot(param_vals, df_results[col_name].to_list(),
                           marker=marker, label=f'MAP@{k}', linewidth=2.5,
                           markersize=8, color=color, alpha=0.8)
    axes[0, 2].axvline(best_val, color='red', linestyle='--', alpha=0.7, linewidth=2)
    axes[0, 2].set_xlabel(param_name, fontsize=12, fontweight='bold')
    axes[0, 2].set_ylabel('MAP Score', fontsize=12, fontweight='bold')
    axes[0, 2].set_title('MAP@K', fontsize=13, fontweight='bold')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].legend(fontsize=10, loc='best')
    
    # ========== ROW 2: RMSE, Accuracy ==========
    
    # RMSE subplot
    if 'RMSE' in df_results.columns:
        axes[1, 0].plot(param_vals, df_results['RMSE'].to_list(),
                       marker='d', linewidth=2.5, markersize=8, color='#E74C3C', alpha=0.8)
        axes[1, 0].axvline(best_val, color='red', linestyle='--', alpha=0.7, linewidth=2)
        axes[1, 0].set_xlabel(param_name, fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('RMSE', fontsize=12, fontweight='bold')
        axes[1, 0].set_title('RMSE (Lower is better)', fontsize=13, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Accuracy subplot
    if 'Accuracy' in df_results.columns:
        axes[1, 1].plot(param_vals, df_results['Accuracy'].to_list(),
                       marker='p', linewidth=2.5, markersize=8, color='#F39C12', alpha=0.8)
        axes[1, 1].axvline(best_val, color='red', linestyle='--', alpha=0.7, linewidth=2)
        axes[1, 1].set_xlabel(param_name, fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        axes[1, 1].set_title('Accuracy', fontsize=13, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
    
    # Hide third subplot in row 2
    axes[1, 2].axis('off')
    
    # ========== ROW 3: All Ranking Metrics Overlay, Summary Table ==========
    
    # All ranking metrics overlay (@20 only for clarity)
    metrics_to_plot = [
        ('NDCG@20', 'o', '#2E86C1', 'NDCG@20'),
        ('Recall@20', 's', '#28B463', 'Recall@20'),
        ('MAP@20', '^', '#9B59B6', 'MAP@20')
    ]
    
    for metric, marker, color, label in metrics_to_plot:
        if metric in df_results.columns:
            axes[2, 0].plot(param_vals, df_results[metric].to_list(),
                           marker=marker, label=label, linewidth=2.5,
                           markersize=8, color=color, alpha=0.8)
    
    axes[2, 0].axvline(best_val, color='red', linestyle='--', alpha=0.7,
                      linewidth=2, label=f'Best={best_val}')
    axes[2, 0].set_xlabel(param_name, fontsize=12, fontweight='bold')
    axes[2, 0].set_ylabel('Score', fontsize=12, fontweight='bold')
    axes[2, 0].set_title('All Ranking Metrics @20', fontsize=13, fontweight='bold')
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].legend(fontsize=10, loc='best')
    
    # Summary table
    create_summary_table(axes[2, 1], df_results, param_col, best_val)
    
    # Hide third subplot in row 3
    axes[2, 2].axis('off')
    
    # Overall title
    plt.suptitle(f'Hyperparameter Tuning Results ({algo_name}) - {category}',
                fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save
    out_path = save_dir / f'{param_col.lower()}_tuning_{category}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return out_path

def visualize_final_results(results_list: list, save_dir: Path, 
                           algo_name: str, k_values: list = [5, 10, 20]):
    """
    Visualize final evaluation results (works for all algorithms)
    
    Args:
        results_list: List of evaluation results
        save_dir: Directory to save plot
        algo_name: Algorithm name for title
        k_values: K values for metrics
    """
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
    """
    Create validation vs test comparison (works for all algorithms)
    
    Args:
        cat: Category name
        param_val: Best parameter value (K or n_factors)
        tuning_row: Validation results row
        final_row: Test results row
        save_dir: Directory to save
        param_name: Parameter name for title
        algo_name: Algorithm name
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    metrics_names = ['NDCG@20', 'Recall@20', 'MAP@20']
    val_scores = [tuning_row['NDCG@20'], tuning_row['Recall@20'], tuning_row['MAP@20']]
    test_scores = [final_row['ndcg@20'], final_row['recall@20'], final_row['map@20']]
    
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
    
    improvements = [f"{(t/v - 1)*100:+.1f}%" if v > 0 else "N/A"
                   for v, t in zip(val_scores, test_scores)]
    
    val_scores_full = val_scores + [tuning_row['RMSE']]
    test_scores_full = test_scores + [final_row['rmse']]
    improvements_full = improvements + [f"{(final_row['rmse']/tuning_row['RMSE'] - 1)*100:+.1f}%"]
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


def visualize_evaluation_results(results_list: list, save_dir: Path, 
                                 algo_name: str, k_values: list = [5, 10, 20]):
    """Visualize final evaluation results (works for all algorithms)"""
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