import os # Used for file path manipulations
from data_split import split_data # Function to split dataset
from decision_tree import CARTDecisionTreeRegressor # CART decision tree regressor
from random_forest import RandomForestRegressor # Random Forest regressor
from hyperparameter_tuning import compare_search_methods # Function to compare hyperparameter tuning methods
from evaluation import evaluate_model, get_feature_importance # Evaluation functions
from visualisation import save_results, save_plots # Functions to save results and plots


def main():
    # Stores all results for saving
    all_results = {}
    
    # 1. Data Splitting
    print("\n1. Data Splitting")
    
    base_dir = os.path.dirname(__file__) # Base directory of the script
    data_path = os.path.join(base_dir, '..', 'Dataset', 'hour.csv') # Path to dataset
    output_dir = os.path.join(base_dir, '..', 'Dataset', 'splits') # Output directory for splits
    
    splits = split_data(data_path, output_dir) # Splits the data and returns dataframes
    feature_names = list(splits['X_train'].columns)
    
    # Stores split information
    all_results['split_info'] = {
        'total_samples': len(splits['X_train']) + len(splits['X_val']) + len(splits['X_test']),
        'train_samples': len(splits['X_train']),
        'val_samples': len(splits['X_val']),
        'test_samples': len(splits['X_test']),
        'n_features': len(feature_names),
        'feature_names': feature_names
    }
    
    # 2. Decision Tree
    print("\n2. Decision Tree")
    
    # Creates and trains the CART decision tree
    dt_model = CARTDecisionTreeRegressor(max_depth=10, min_samples_leaf=5)
    dt_model.fit(splits['X_train'], splits['y_train'])
    
    # Evaluates on all datasets
    print("\nDecision Tree Evaluation")
    dt_train = evaluate_model(dt_model, splits['X_train'], splits['y_train'], "Training set")
    dt_val = evaluate_model(dt_model, splits['X_val'], splits['y_val'], "Validation set")
    dt_test = evaluate_model(dt_model, splits['X_test'], splits['y_test'], "Test set")
    
    # Gets feature importances
    print("\nDecision Tree Feature Importance")
    dt_importance = get_feature_importance(dt_model, feature_names)
    
    # Stores decision tree results
    all_results['decision_tree'] = {
        'train': dt_train,
        'val': dt_val,
        'test': dt_test,
        'feature_importance': dt_importance
    }
    
    # 3. Random Forest
    print("\n3. Random Forest")
    
    # Creates and trains the random forest with predefined hyperparameters
    rf_model = RandomForestRegressor(
        n_estimators=50,
        max_depth=10,
        min_samples_leaf=5,
        max_features='sqrt',
        random_state=50
    )
    rf_model.fit(splits['X_train'], splits['y_train'])
    
    # Evaluates on all datasets
    print("\nRandom Forest Evaluation")
    rf_train = evaluate_model(rf_model, splits['X_train'], splits['y_train'], "Training set")
    rf_val = evaluate_model(rf_model, splits['X_val'], splits['y_val'], "Validation set")
    rf_test = evaluate_model(rf_model, splits['X_test'], splits['y_test'], "Test set")
    
    # Gets feature importances
    print("\nRandom Forest Feature Importance")
    rf_importance = get_feature_importance(rf_model, feature_names)
    
    # Stores random forest results
    all_results['random_forest'] = {
        'train': rf_train,
        'val': rf_val,
        'test': rf_test,
        'feature_importance': rf_importance
    }
    
    # 4. Random Forest Hyperparameter Tuning
    print("\n4. Random Forest Hyperparameter Tuning")
    
    # Defines parameter grid for RF and tuning
    rf_param_grid = {
        'n_estimators': [10, 30],
        'max_depth': [5, 10],
        'min_samples_leaf': [2, 5, 10],
        'criterion': ['mse', 'mae', 'ssr']
    }
    
    print("\nParameter grid:")
    for param, values in rf_param_grid.items():
        print(f"  {param}: {values}")
    
    tuning_results = compare_search_methods(
        param_grid=rf_param_grid,
        X_train=splits['X_train'],
        y_train=splits['y_train'],
        X_test=splits['X_test'],
        y_test=splits['y_test'],
        n_iter_random=10,
        cv=3,
        scoring='r2',
        random_state=50
    )
    
    # Stores tuning results
    all_results['tuning'] = {
        'grid_search': {
            'best_score': tuning_results['grid_cv'].best_score_,
            'best_params': tuning_results['grid_cv'].best_params_
        },
        'random_search': {
            'best_score': tuning_results['random_cv'].best_score_,
            'best_params': tuning_results['random_cv'].best_params_
        },
        'grid_test_metrics': tuning_results['grid_test_metrics'],
        'random_test_metrics': tuning_results['random_test_metrics']
    }
    
    # 5. Final Summary
    print("\n5. Final Summary")
    
    print(f"\n{'Model':<30} {'Test R²':<12} {'Test RMSE':<12} {'Test MAE':<12}")
    print(f"{'Decision Tree':<30} {dt_test['r2']:<12.4f} {dt_test['rmse']:<12.2f} {dt_test['mae']:<12.2f}")
    print(f"{'Random Forest':<30} {rf_test['r2']:<12.4f} {rf_test['rmse']:<12.2f} {rf_test['mae']:<12.2f}")
    print(f"{'RF Tuned (Grid Search)':<30} {tuning_results['grid_test_metrics']['r2']:<12.4f} {tuning_results['grid_test_metrics']['rmse']:<12.2f} {tuning_results['grid_test_metrics']['mae']:<12.2f}")
    print(f"{'RF Tuned (Random Search)':<30} {tuning_results['random_test_metrics']['r2']:<12.4f} {tuning_results['random_test_metrics']['rmse']:<12.2f} {tuning_results['random_test_metrics']['mae']:<12.2f}")
    
    print("\n--- Best Hyperparameters ---")
    print("\nGrid Search:")
    for param in sorted(tuning_results['grid_cv'].best_params_.keys()):
        print(f"  {param}: {tuning_results['grid_cv'].best_params_[param]}")
    
    print("\nRandom Search:")
    for param in sorted(tuning_results['random_cv'].best_params_.keys()):
        print(f"  {param}: {tuning_results['random_cv'].best_params_[param]}")
    
    # 6. Save Results
    results_dir = os.path.join(base_dir, '..', 'results')
    save_results(results_dir, all_results)
    
    # 7. Generate Plots
    save_plots(results_dir, all_results, splits['y_test'])
    
    print("\nCompleted")


if __name__ == "__main__":
    main()
