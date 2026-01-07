import os # Used to make directories and handle file paths
import numpy as np # For numerical operations
import matplotlib.pyplot as plt # For generating plots

"""
Saves all results to files in the specified output directory.

Parameters:
    output_dir: Directory to save results
    results: Dictionary containing all results to save
"""
def save_results(output_dir, results):
    # Creates output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nSaving Results")
    
    # Saves hyperparameter tuning results
    save_tuning_results(output_dir, results.get('tuning', {}))
    
    print(f"\nAll results saved to: {output_dir}")


"""
Saves hyperparameter tuning results.

Parameters:
    output_dir: Directory to save results
    tuning_results: Dictionary with tuning results
"""
def save_tuning_results(output_dir, tuning_results):
    if not tuning_results:
        return
    
    filepath = os.path.join(output_dir, 'hyperparameter_tuning.txt')
    
    with open(filepath, 'w') as f:
        f.write("Hyperparameter Tuning Results\n")
        
        # Grid Search results
        if 'grid_search' in tuning_results:
            f.write("Grid Search\n")
            f.write(f"Best CV Score: {tuning_results['grid_search'].get('best_score', 'N/A'):.4f}\n")
            f.write("Best Parameters:\n")
            for param, value in tuning_results['grid_search'].get('best_params', {}).items():
                f.write(f"  {param}: {value}\n")
            f.write("\nTest Set Performance:\n")
            test_metrics = tuning_results.get('grid_test_metrics', {})
            f.write(f"  R^2: {test_metrics.get('r2', 0):.4f}\n")
            f.write(f"  RMSE: {test_metrics.get('rmse', 0):.2f}\n")
            f.write(f"  MAE: {test_metrics.get('mae', 0):.2f}\n")
            f.write("\n")
        
        # Random Search results
        if 'random_search' in tuning_results:
            f.write("Random Search\n")
            f.write(f"Best CV Score: {tuning_results['random_search'].get('best_score', 'N/A'):.4f}\n")
            f.write("Best Parameters:\n")
            for param, value in tuning_results['random_search'].get('best_params', {}).items():
                f.write(f"  {param}: {value}\n")
            f.write("\nTest Set Performance:\n")
            test_metrics = tuning_results.get('random_test_metrics', {})
            f.write(f"  R^2: {test_metrics.get('r2', 0):.4f}\n")
            f.write(f"  RMSE: {test_metrics.get('rmse', 0):.2f}\n")
            f.write(f"  MAE: {test_metrics.get('mae', 0):.2f}\n")
    
    print(f"  Saved: hyperparameter_tuning.txt")


"""
Generates and saves visualisation plots.

Parameters:
    output_dir: Directory to save plots
    results: Dictionary containing all results
    y_test: Actual test target values
"""
def save_plots(output_dir, results, y_test):
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    print("\nGenerating Plots")
    
    # 1. Feature Importance Comparison
    save_feature_importance_plot(plots_dir, results)
    
    # 2. Predictions vs Actual
    save_predictions_plot(plots_dir, results, y_test)
    
    # 3. Model Performance Comparison
    save_model_comparison_plot(plots_dir, results)
    
    print(f"  Plots saved to: {plots_dir}")


"""
Saves feature importance comparison bar chart.
"""
def save_feature_importance_plot(plots_dir, results):
    _, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, (model_name, title) in enumerate([('decision_tree', 'Decision Tree'), 
                                                ('random_forest', 'Random Forest')]):
        if model_name in results and 'feature_importance' in results[model_name]:
            importance = results[model_name]['feature_importance']
            # Sort by importance
            sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
            features, values = zip(*sorted_features)
            
            axes[idx].barh(range(len(features)), values, color='steelblue')
            axes[idx].set_yticks(range(len(features)))
            axes[idx].set_yticklabels(features)
            axes[idx].invert_yaxis()
            axes[idx].set_xlabel('Importance')
            axes[idx].set_title(f'{title} - Top 10 Features')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'feature_importance.png'), dpi=150)
    plt.close()
    print("  Saved: feature_importance.png")


"""
Saves predictions vs actual scatter plot.
"""
def save_predictions_plot(plots_dir, results, y_test):
    _, axes = plt.subplots(1, 2, figsize=(12, 5))
    y_test = np.array(y_test)
    
    for idx, (model_name, title) in enumerate([('decision_tree', 'Decision Tree'), 
                                                ('random_forest', 'Random Forest')]):
        if model_name in results and 'test' in results[model_name]:
            y_pred = results[model_name]['test'].get('predictions', [])
            if len(y_pred) > 0:
                axes[idx].scatter(y_test, y_pred, alpha=0.3, s=10, color='steelblue')
                # Perfect prediction line
                min_val = min(y_test.min(), np.min(y_pred))
                max_val = max(y_test.max(), np.max(y_pred))
                axes[idx].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction') # r-- is red dashed line
                axes[idx].set_xlabel('Actual')
                axes[idx].set_ylabel('Predicted')
                axes[idx].set_title(f'{title} (R² = {results[model_name]["test"]["r2"]:.4f})')
                axes[idx].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'predictions_vs_actual.png'), dpi=150)
    plt.close()
    print("  Saved: predictions_vs_actual.png")


"""
Saves model performance comparison bar chart.
"""
def save_model_comparison_plot(plots_dir, results):
    models = []
    r2_scores = []
    rmse_scores = []
    
    # Collect metrics
    if 'decision_tree' in results and 'test' in results['decision_tree']:
        models.append('Decision Tree')
        r2_scores.append(results['decision_tree']['test']['r2'])
        rmse_scores.append(results['decision_tree']['test']['rmse'])
    
    if 'random_forest' in results and 'test' in results['random_forest']:
        models.append('Random Forest')
        r2_scores.append(results['random_forest']['test']['r2'])
        rmse_scores.append(results['random_forest']['test']['rmse'])
    
    if 'tuning' in results:
        if 'grid_test_metrics' in results['tuning']:
            models.append('RF (Grid Search)')
            r2_scores.append(results['tuning']['grid_test_metrics']['r2'])
            rmse_scores.append(results['tuning']['grid_test_metrics']['rmse'])
        if 'random_test_metrics' in results['tuning']:
            models.append('RF (Random Search)')
            r2_scores.append(results['tuning']['random_test_metrics']['r2'])
            rmse_scores.append(results['tuning']['random_test_metrics']['rmse'])
    
    if not models:
        return
    
    _, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = range(len(models))
    
    # R² Score
    axes[0].bar(x, r2_scores, color='steelblue')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=15, ha='right') # Rotates x-axis labels for readability
    axes[0].set_ylabel('R² Score')
    axes[0].set_title('Model Comparison - R² Score')
    axes[0].set_ylim(0, 1)
    
    # RMSE
    axes[1].bar(x, rmse_scores, color='coral')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=15, ha='right')
    axes[1].set_ylabel('RMSE')
    axes[1].set_title('Model Comparison - RMSE')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'model_comparison.png'), dpi=150)
    plt.close()
    print("Saved: model_comparison.png")
