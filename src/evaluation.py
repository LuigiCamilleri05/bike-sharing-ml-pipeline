import numpy as np # For numerical operations


"""
Evaluates the model and returns performance metrics.
    
Parameters:
    model: Trained model with predict() method
    X: Features
    y: True target values
    dataset_name: Name for display purposes
    
Returns:
    Dictionary with evaluation metrics
"""
def evaluate_model(model, X, y, dataset_name):
    
    # Makes predictions
    y_pred = model.predict(X)
    y = np.array(y)
    
    # Calculates metrics
    mse = np.mean((y - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y - y_pred))
    
    # Calculates R² score
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # Displays results
    print(f"\n{dataset_name} performance:")
    print(f"  MSE:  {mse:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE:  {mae:.2f}")
    print(f"  R²:   {r2:.4f}")
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'predictions': y_pred
    }


"""
Retrieves and displays feature importances from the trained model.
    
Parameters:
    model: Trained model with feature_importances_ attribute
    feature_names: List of feature names
    
Returns:
    Dictionary mapping feature names to importance scores
"""
def get_feature_importance(model, feature_names):
    
    importances = model.feature_importances_
    
    # Creates sorted list of (feature, importance) tuples
    feature_importance = list(zip(feature_names, importances))
    feature_importance.sort(key=lambda x: x[1], reverse=True) # Sorts by importance
    
    print("\nFeature importances:")
    for feature, importance in feature_importance:
        if importance > 0:
            print(f"  {feature}: {importance:.4f}")
    
    return dict(feature_importance)
