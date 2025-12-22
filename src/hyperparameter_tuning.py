import time
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from random_forest import RandomForestRegressor
from evaluation import evaluate_model


"""
Performs Grid Search using GridSearchCV with Random Forest.

Parameters:
    param_grid: Dictionary of parameter names and their possible values
    X_train: Training features
    y_train: Training targets
    cv: Number of cross-validation folds
    scoring: Scoring metric ('r2', 'neg_mean_squared_error', etc.)
    verbose: Verbosity level 

Returns:
    Fitted GridSearchCV object
"""
def grid_search(param_grid, X_train, y_train, cv=3, scoring='r2', verbose=1):
    
    print(f"\nGrid Search: Using RandomForestRegressor")
    print(f"Cross-validation folds: {cv}")
    
    # Creates base model
    base_model = RandomForestRegressor(random_state=50)
    
    # Creates GridSearchCV with RF
    grid_cv = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=1,  # Sequential due to custom model
        verbose=verbose,
        refit=True
    )
    
    # Times the search
    start_time = time.time()
    grid_cv.fit(X_train, y_train)
    search_time = time.time() - start_time
    
    print(f"\nGrid Search completed in {search_time:.2f} seconds")
    print(f"Best score (CV): {grid_cv.best_score_:.4f}")
    print(f"Best parameters: {grid_cv.best_params_}")
    
    return grid_cv


"""
Performs Random Search using RandomizedSearchCV with Random Forest.

Parameters:
    param_distributions: Dictionary of parameter names and their possible values/distributions
    X_train: Training features
    y_train: Training targets
    n_iter: Number of parameter settings sampled
    cv: Number of cross-validation folds
    scoring: Scoring metric
    random_state: Random seed
    verbose: Verbosity level

Returns:
    Fitted RandomizedSearchCV object
"""
def random_search(param_distributions, X_train, y_train, n_iter=10, cv=3, 
                  scoring='r2', random_state=50, verbose=1):
    
    print(f"\nRandom Search: Using RandomForestRegressor")
    print(f"Iterations: {n_iter}, CV folds: {cv}")
    
    # Creates base model
    base_model = RandomForestRegressor(random_state=50)
    
    # Creates RandomizedSearchCV with RF
    random_cv = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        random_state=random_state,
        n_jobs=1,  # Sequential due to custom model
        verbose=verbose,
        refit=True
    )
    
    start_time = time.time()
    random_cv.fit(X_train, y_train)
    search_time = time.time() - start_time
    
    print(f"\nRandom Search completed in {search_time:.2f} seconds")
    print(f"Best score (CV): {random_cv.best_score_:.4f}")
    print(f"Best parameters: {random_cv.best_params_}")
    
    return random_cv


"""
Compares Grid Search and Random Search performance

Parameters:
    param_grid: Dictionary of parameter names and their possible values
    X_train: Training features
    y_train: Training targets
    X_test: Test features
    y_test: Test targets
    n_iter_random: Number of iterations for random search
    cv: Number of cross-validation folds
    scoring: Scoring metric
    random_state: Random seed

Returns:
    Dictionary with comparison results
"""
def compare_search_methods(param_grid, X_train, y_train, X_test, y_test,
                           n_iter_random=10, cv=3, scoring='r2', random_state=50):
    
    print("\nHyperparameter Tuning Comparison")
    print("Using GridSearchCV/RandomizedSearchCV with RF")
    
    # Calculates total combinations
    n_combinations = 1
    for values in param_grid.values():
        n_combinations *= len(values)
    print(f"\nTotal parameter combinations: {n_combinations}")
    
    # Performs Grid Search
    print("\n--- Grid Search ---")
    grid_cv = grid_search(
        param_grid, X_train, y_train,
        cv=cv, scoring=scoring, verbose=1
    )
    
    # Performs Random Search
    print("\n--- Random Search ---")
    random_cv = random_search(
        param_grid, X_train, y_train,
        n_iter=n_iter_random, cv=cv, scoring=scoring, 
        random_state=random_state, verbose=1
    )
    
    # Evaluates best models on test set
    print("\nFinal Model Evaluation on Test Set")
    
    print("\n--- Grid Search Best Model ---")
    grid_test_metrics = evaluate_model(grid_cv.best_estimator_, X_test, y_test, "Test")
    
    print("\n--- Random Search Best Model ---")
    random_test_metrics = evaluate_model(random_cv.best_estimator_, X_test, y_test, "Test")
    
    # Prints comparison summary
    print("\nSearch Method Comparison Summary")
    
    print(f"\n{'Metric':<30} {'Grid Search':<20} {'Random Search':<20}")
    print("-" * 70)
    print(f"{'Combinations evaluated':<30} {n_combinations:<20} {n_iter_random:<20}")
    print(f"{'Best CV score':<30} {grid_cv.best_score_:<20.4f} {random_cv.best_score_:<20.4f}")
    print(f"{'Test R²':<30} {grid_test_metrics['r2']:<20.4f} {random_test_metrics['r2']:<20.4f}")
    print(f"{'Test RMSE':<30} {grid_test_metrics['rmse']:<20.4f} {random_test_metrics['rmse']:<20.4f}")
    
    print("\nBest Parameters (Grid Search):")
    for param, value in grid_cv.best_params_.items():
        print(f"  {param}: {value}")
    
    print("\nBest Parameters (Random Search):")
    for param, value in random_cv.best_params_.items():
        print(f"  {param}: {value}")
    
    return {
        'grid_cv': grid_cv,
        'random_cv': random_cv,
        'grid_test_metrics': grid_test_metrics,
        'random_test_metrics': random_test_metrics
    }
