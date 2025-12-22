import os
from data_split import split_data


def main():
    # defined paths
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, '..', 'Dataset', 'hour.csv')
    output_dir = os.path.join(base_dir, '..', 'Dataset', 'splits')

    # splits the data
    # the splits dictionary contains:
    # - X_train, X_val, X_test: feature dataframes
    # - y_train, y_val, y_test: target series
    splits = split_data(data_path, output_dir)

    print(f"Training samples: {len(splits['X_train'])}")
    print(f"Validation samples: {len(splits['X_val'])}")
    print(f"Test samples: {len(splits['X_test'])}")


if __name__ == "__main__":
    main()
