"""
Visualisation module for saving model results to files.
"""

import os
import json
import numpy as np
import pandas as pd


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
    
    # Saves data split summary
    save_split_summary(output_dir, results.get('split_info', {}))
    
    # Saves decision tree results
    save_model_results(output_dir, 'decision_tree', results.get('decision_tree', {}))
    
    # Saves random forest results
    save_model_results(output_dir, 'random_forest', results.get('random_forest', {}))
    
    # Saves hyperparameter tuning results
    save_tuning_results(output_dir, results.get('tuning', {}))
    
    # Saves comparison summary
    save_comparison_summary(output_dir, results)
    
    print(f"\nAll results saved to: {output_dir}")


"""
Saves data split summary to file.

Parameters:
    output_dir: Directory to save results
    split_info: Dictionary with split information
"""
def save_split_summary(output_dir, split_info):
    filepath = os.path.join(output_dir, 'data_split_summary.txt')
    
    with open(filepath, 'w') as f:
        f.write("DATA SPLIT SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total samples: {split_info.get('total_samples', 'N/A')}\n")
        f.write(f"Training samples: {split_info.get('train_samples', 'N/A')}\n")
        f.write(f"Validation samples: {split_info.get('val_samples', 'N/A')}\n")
        f.write(f"Test samples: {split_info.get('test_samples', 'N/A')}\n")
        f.write(f"Number of features: {split_info.get('n_features', 'N/A')}\n")
        f.write(f"\nFeatures: {split_info.get('feature_names', [])}\n")
    
    print(f"  Saved: data_split_summary.txt")


"""
Saves model results to files.

Parameters:
    output_dir: Directory to save results
    model_name: Name of the model
    model_results: Dictionary with model results
"""
def save_model_results(output_dir, model_name, model_results):
    if not model_results:
        return
    
    # Saves metrics
    metrics_path = os.path.join(output_dir, f'{model_name}_metrics.csv')
    metrics_data = []
    
    for dataset_name in ['train', 'val', 'test']:
        if dataset_name in model_results:
            metrics = model_results[dataset_name]
            metrics_data.append({
                'Dataset': dataset_name.capitalize(),
                'MSE': metrics.get('mse', 0),
                'RMSE': metrics.get('rmse', 0),
                'MAE': metrics.get('mae', 0),
                'R2': metrics.get('r2', 0)
            })
    
    if metrics_data:
        df = pd.DataFrame(metrics_data)
        df.to_csv(metrics_path, index=False)
        print(f"  Saved: {model_name}_metrics.csv")
    
    # Saves feature importances
    if 'feature_importance' in model_results:
        importance_path = os.path.join(output_dir, f'{model_name}_feature_importance.csv')
        importance_df = pd.DataFrame(
            list(model_results['feature_importance'].items()),
            columns=['Feature', 'Importance']
        )
        importance_df = importance_df.sort_values('Importance', ascending=False)
        importance_df.to_csv(importance_path, index=False)
        print(f"  Saved: {model_name}_feature_importance.csv")


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
        f.write("HYPERPARAMETER TUNING RESULTS\n")
        f.write("=" * 50 + "\n\n")
        
        # Grid Search results
        if 'grid_search' in tuning_results:
            f.write("GRID SEARCH\n")
            f.write("-" * 30 + "\n")
            f.write(f"Best CV Score: {tuning_results['grid_search'].get('best_score', 'N/A'):.4f}\n")
            f.write("Best Parameters:\n")
            for param, value in tuning_results['grid_search'].get('best_params', {}).items():
                f.write(f"  {param}: {value}\n")
            f.write("\nTest Set Performance:\n")
            test_metrics = tuning_results.get('grid_test_metrics', {})
            f.write(f"  R²: {test_metrics.get('r2', 0):.4f}\n")
            f.write(f"  RMSE: {test_metrics.get('rmse', 0):.2f}\n")
            f.write(f"  MAE: {test_metrics.get('mae', 0):.2f}\n")
            f.write("\n")
        
        # Random Search results
        if 'random_search' in tuning_results:
            f.write("RANDOM SEARCH\n")
            f.write("-" * 30 + "\n")
            f.write(f"Best CV Score: {tuning_results['random_search'].get('best_score', 'N/A'):.4f}\n")
            f.write("Best Parameters:\n")
            for param, value in tuning_results['random_search'].get('best_params', {}).items():
                f.write(f"  {param}: {value}\n")
            f.write("\nTest Set Performance:\n")
            test_metrics = tuning_results.get('random_test_metrics', {})
            f.write(f"  R²: {test_metrics.get('r2', 0):.4f}\n")
            f.write(f"  RMSE: {test_metrics.get('rmse', 0):.2f}\n")
            f.write(f"  MAE: {test_metrics.get('mae', 0):.2f}\n")
    
    print(f"  Saved: hyperparameter_tuning.txt")
    
    # Saves best parameters as JSON
    best_params_path = os.path.join(output_dir, 'best_hyperparameters.json')
    best_params = {
        'grid_search': tuning_results.get('grid_search', {}).get('best_params', {}),
        'random_search': tuning_results.get('random_search', {}).get('best_params', {})
    }
    
    # Converts numpy types to Python types for JSON serialisation
    best_params = _convert_to_serialisable(best_params)
    
    with open(best_params_path, 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f"  Saved: best_hyperparameters.json")


"""
Saves comparison summary of all models.

Parameters:
    output_dir: Directory to save results
    results: Dictionary containing all results
"""
def save_comparison_summary(output_dir, results):
    filepath = os.path.join(output_dir, 'model_comparison.csv')
    
    comparison_data = []
    
    # Decision Tree
    if 'decision_tree' in results and 'test' in results['decision_tree']:
        dt_test = results['decision_tree']['test']
        comparison_data.append({
            'Model': 'Decision Tree',
            'Test_R2': dt_test.get('r2', 0),
            'Test_RMSE': dt_test.get('rmse', 0),
            'Test_MAE': dt_test.get('mae', 0)
        })
    
    # Random Forest
    if 'random_forest' in results and 'test' in results['random_forest']:
        rf_test = results['random_forest']['test']
        comparison_data.append({
            'Model': 'Random Forest',
            'Test_R2': rf_test.get('r2', 0),
            'Test_RMSE': rf_test.get('rmse', 0),
            'Test_MAE': rf_test.get('mae', 0)
        })
    
    # Tuned Random Forest (Grid Search)
    if 'tuning' in results and 'grid_test_metrics' in results['tuning']:
        grid_test = results['tuning']['grid_test_metrics']
        comparison_data.append({
            'Model': 'RF Tuned (Grid Search)',
            'Test_R2': grid_test.get('r2', 0),
            'Test_RMSE': grid_test.get('rmse', 0),
            'Test_MAE': grid_test.get('mae', 0)
        })
    
    # Tuned Random Forest (Random Search)
    if 'tuning' in results and 'random_test_metrics' in results['tuning']:
        random_test = results['tuning']['random_test_metrics']
        comparison_data.append({
            'Model': 'RF Tuned (Random Search)',
            'Test_R2': random_test.get('r2', 0),
            'Test_RMSE': random_test.get('rmse', 0),
            'Test_MAE': random_test.get('mae', 0)
        })
    
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        df.to_csv(filepath, index=False)
        print(f"  Saved: model_comparison.csv")


"""
Converts numpy types to Python types for JSON serialisation.

Parameters:
    obj: Object to convert

Returns:
    Converted object
"""
def _convert_to_serialisable(obj):
    if isinstance(obj, dict):
        return {k: _convert_to_serialisable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_serialisable(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj
