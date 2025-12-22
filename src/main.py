import os
from data_split import split_data
from decision_tree import create_decision_tree, train_decision_tree, evaluate_model, get_feature_importance


def main():
    # defined paths
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, '..', 'Dataset', 'hour.csv')
    output_dir = os.path.join(base_dir, '..', 'Dataset', 'splits')

    # split the data
    splits = split_data(data_path, output_dir)

    print("\n" + "=" * 50)
    print("decision tree training")
    print("=" * 50)

    # created and trained the CART decision tree
    model = create_decision_tree(max_depth=10, min_samples_leaf=5)
    model = train_decision_tree(model, splits['X_train'], splits['y_train'])

    # evaluated on all datasets
    print("\n" + "=" * 50)
    print("model evaluation")
    print("=" * 50)
    
    train_metrics = evaluate_model(model, splits['X_train'], splits['y_train'], "training set")
    val_metrics = evaluate_model(model, splits['X_val'], splits['y_val'], "validation set")
    test_metrics = evaluate_model(model, splits['X_test'], splits['y_test'], "test set")

    # displayed feature importances
    print("\n" + "=" * 50)
    print("feature analysis")
    print("=" * 50)
    feature_names = list(splits['X_train'].columns)
    importance = get_feature_importance(model, feature_names)


if __name__ == "__main__":
    main()
