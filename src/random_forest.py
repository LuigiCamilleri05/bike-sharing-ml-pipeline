import numpy as np # Used for numerical operations
from decision_tree import CARTDecisionTreeRegressor # Importing the CART decision tree implementation

"""
Random forest regressor implemented from scratch.
Uses bootstrap aggregating (bagging) and random feature selection at each split.
Initialises the random forest with hyperparameters.
        
Parameters:
    n_estimators: Number of trees in the forest
    max_depth: Maximum depth of each tree
    min_samples_split: Minimum samples required to split a node
    min_samples_leaf: Minimum samples required in a leaf node
    max_features: Number of features to consider at each split
        - 'sqrt': sqrt(n_features)
        - 'log2': log2(n_features)
        - None: all features
    criterion: Splitting criterion - 'mse', 'mae', or 'ssr'
    random_state: Random seed for reproducibility
"""
class RandomForestRegressor:

    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, 
                 min_samples_leaf=1, max_features='sqrt', criterion='mse', random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.criterion = criterion
        self.random_state = random_state
        
        self.trees = []
        self.feature_indices = []  # Stores which features each tree uses
        self.n_features = None
        self.feature_importances_ = None

    """
    Builds the random forest from training data.
        
    Parameters:
        X: Training features (numpy array or pandas DataFrame)
        y: Training target (numpy array or pandas Series)
    """
    def fit(self, X, y):
        # Converts to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        n_samples, self.n_features = X.shape
        
        # Sets random seed for reproducibility
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Initialises feature importances
        self.feature_importances_ = np.zeros(self.n_features)
        
        self.trees = []
        self.feature_indices = []
        
        print(f"Training random forest with {self.n_estimators} trees...")
        
        for i in range(self.n_estimators):
            # Creates bootstrap sample (sampling with replacement)
            bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_bootstrap = X[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]
            
            # Stores feature indices (all features since selection happens at each split)
            feature_idx = np.arange(self.n_features)
            self.feature_indices.append(feature_idx)
            
            # Creates and trains tree on bootstrap sample
            # Random feature selection at each split is handled by max_features parameter
            tree = CARTDecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                criterion=self.criterion,
                max_features=self._get_max_features()
            )
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)
            
            # Accumulates feature importances across all trees
            self.feature_importances_ += tree.feature_importances_
            
            # Prints progress every 10 trees
            if (i + 1) % 10 == 0:
                print(f"  Trained {i + 1}/{self.n_estimators} trees")
        
        # Normalises feature importances to sum to 1
        if self.feature_importances_.sum() > 0:
            self.feature_importances_ /= self.feature_importances_.sum()
        
        print(f"Completed training {self.n_estimators} trees")
        
        return self
    
    """
    Determines the number of features to consider at each split.
        
    Returns:
        Number of features
    """
    def _get_max_features(self):
        if self.max_features is None:
            return self.n_features
        elif self.max_features == 'sqrt':
            return max(1, int(np.sqrt(self.n_features)))
        elif self.max_features == 'log2':
            return max(1, int(np.log2(self.n_features)))
        else:
            return self.n_features
    
    """
    Predicts target values by averaging predictions from all trees.
        
    Parameters:
        X: Feature matrix
        
    Returns:
        Array of predictions
    """
    def predict(self, X):
        X = np.array(X)
        
        # Collects predictions from all trees
        all_predictions = []
        
        for tree, feature_idx in zip(self.trees, self.feature_indices):
            # Uses only the features this tree was trained on
            tree_pred = tree.predict(X[:, feature_idx])
            all_predictions.append(tree_pred)
        
        # Averages predictions across all trees
        all_predictions = np.array(all_predictions)
        return np.mean(all_predictions, axis=0)
    
    """
    Gets parameters for this estimator.
    Needed to use GridSearchCV/RandomizedSearchCV.
    
    Parameters:
        deep: If True, returns parameters for sub-objects
        
    Returns:
        Dictionary of parameter names and values
    """
    def get_params(self, deep=True):
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'max_features': self.max_features,
            'criterion': self.criterion,
            'random_state': self.random_state
        }
    
    """
    Sets parameters for this estimator (sklearn-compatible).
    Required for sklearn's GridSearchCV/RandomizedSearchCV.
    
    Parameters:
        **params: Dictionary of parameters to set
        
    Returns:
        Self
    """
    def set_params(self, **params):
        for param, value in params.items():
            if hasattr(self, param):
                setattr(self, param, value)
        return self


"""
Creates a random forest regressor with specified hyperparameters.
    
Parameters:
    n_estimators: Number of trees in the forest
    max_depth: Maximum depth of each tree
    min_samples_split: Minimum samples required to split a node
    min_samples_leaf: Minimum samples required in a leaf node
    max_features: Features to consider at each split ('sqrt', 'log2', None)
    criterion: Splitting criterion - 'mse', 'mae', or 'ssr'
    random_state: Random seed for reproducibility
    
Returns:
    RandomForestRegressor instance
"""
def create_random_forest(n_estimators=100, max_depth=None, min_samples_split=2,
                         min_samples_leaf=1, max_features='sqrt', criterion='mse', 
                         random_state=50):
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        criterion=criterion,
        random_state=random_state
    )
    return model
