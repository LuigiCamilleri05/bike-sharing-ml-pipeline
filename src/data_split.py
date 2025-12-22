import pandas as pd
from sklearn.model_selection import train_test_split
import os

"""
Splits the bike sharing dataset into training, validation, and test sets.

Parameters:
    data_path: Path to the hour.csv file
    output_dir: Directory to save the split files
    test_size: Proportion of data for validation + test (default 0.30)
    val_ratio: Ratio to split temp into validation and test (default 0.50)
    random_state: Random seed for reproducibility
    
Returns:
    Dictionary containing the split dataframes and statistics
"""
def split_data(data_path, output_dir, test_size=0.30, val_ratio=0.50, random_state=50):
    
    # Loads the dataset
    df = pd.read_csv(data_path)

    print("=" * 50)
    print("Data splitting")
    print("=" * 50)
    print(f"\nOriginal dataset shape: {df.shape}")
    print(f"Total samples: {len(df)}")

    # Displays basic info about the dataset
    print(f"\nFeatures: {list(df.columns)}")
    print(f"Target variable: 'cnt'")

    # Defines features (X) and target (y)
    # Drops non-predictive columns: instant (index), dteday (date string), casual, registered
    # casual + registered = cnt, so we keep only cnt as target
    X = df.drop(columns=['instant', 'dteday', 'casual', 'registered', 'cnt'])
    y = df['cnt']

    print(f"\nFeature columns used: {list(X.columns)}")
    print(f"Number of features: {X.shape[1]}")

    # First split: 70% training, 30% temporary (becomes validation + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )

    # Second split: Splits the 30% into 15% validation and 15% test (50-50 of the temp)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state, shuffle=True
    )

    # Saves the splits to CSV files
    os.makedirs(output_dir, exist_ok=True)

    # Combines X and y for saving
    train_df = X_train.copy()
    train_df['cnt'] = y_train

    val_df = X_val.copy()
    val_df['cnt'] = y_val

    test_df = X_test.copy()
    test_df['cnt'] = y_test

    # Saves to CSV
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'validation.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)

    print("\nData splitting complete")

    return {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
        'train_df': train_df, 'val_df': val_df, 'test_df': test_df
    }
