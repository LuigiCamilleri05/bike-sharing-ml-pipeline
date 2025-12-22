import os
from data_split import split_data
from decision_tree import create_decision_tree, train_decision_tree, evaluate_model, get_feature_importance
from random_forest import create_random_forest, train_random_forest


def main():
    # Defines paths
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, '..', 'Dataset', 'hour.csv')
    output_dir = os.path.join(base_dir, '..', 'Dataset', 'splits')

    # Splits the data
    splits = split_data(data_path, output_dir)
    feature_names = list(splits['X_train'].columns)

    # Decision Tree
    print("\n" + "=" * 50)
    print("DECISION TREE (CART)")
    print("=" * 50)

    # Creates and trains the CART decision tree
    dt_model = create_decision_tree(max_depth=10, min_samples_leaf=5)
    dt_model = train_decision_tree(dt_model, splits['X_train'], splits['y_train'])

    # Evaluates on all datasets
    print("\n--- Decision Tree Evaluation ---")
    dt_train = evaluate_model(dt_model, splits['X_train'], splits['y_train'], "Training set")
    dt_val = evaluate_model(dt_model, splits['X_val'], splits['y_val'], "Validation set")
    dt_test = evaluate_model(dt_model, splits['X_test'], splits['y_test'], "Test set")

    # Displays feature importances
    print("\n--- Decision Tree Feature Importance ---")
    dt_importance = get_feature_importance(dt_model, feature_names)

    # Random forest
    print("\n" + "=" * 50)
    print("Random forest")
    print("=" * 50)

    # Creates and trains the random forest
    rf_model = create_random_forest(
        n_estimators=50,
        max_depth=10,
        min_samples_leaf=5,
        max_features='sqrt',
        random_state=50
    )
    rf_model = train_random_forest(rf_model, splits['X_train'], splits['y_train'])

    # Evaluates on all datasets (reuses evaluate_model from decision_tree.py)
    print("\n--- Random Forest Evaluation ---")
    rf_train = evaluate_model(rf_model, splits['X_train'], splits['y_train'], "Training set")
    rf_val = evaluate_model(rf_model, splits['X_val'], splits['y_val'], "Validation set")
    rf_test = evaluate_model(rf_model, splits['X_test'], splits['y_test'], "Test set")

    # Displays feature importances (reuses get_feature_importance from decision_tree.py)
    print("\n--- Random Forest Feature Importance ---")
    rf_importance = get_feature_importance(rf_model, feature_names)

    print("\n" + "=" * 50)
    print("Model Comparison (Test Set)")
    print("=" * 50)
    print(f"\n{'Model':<20} {'RMSE':<10} {'MAE':<10} {'R²':<10}")
    print("-" * 50)
    print(f"{'Decision Tree':<20} {dt_test['rmse']:<10.2f} {dt_test['mae']:<10.2f} {dt_test['r2']:<10.4f}")
    print(f"{'Random Forest':<20} {rf_test['rmse']:<10.2f} {rf_test['mae']:<10.2f} {rf_test['r2']:<10.4f}")


if __name__ == "__main__":
    main()
