"""
CART decision tree implementation from scratch.
"""

# Used for numerical computations
import numpy as np 

"""
Represents a node in the decision tree.
"""
class Node:

    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        # For decision nodes
        self.feature = feature      # Index of feature to split on
        self.threshold = threshold  # Threshold value for the split
        self.left = left            # Left child node (<=threshold)
        self.right = right          # Right child node (>threshold)

        # For leaf nodes
        self.value = value          # Predicted value
    
    def is_leaf(self):
        """Checks if this node is a leaf node."""
        return self.value is not None

"""
CART decision tree regressor implemented from scratch.
Supports multiple splitting criteria: MSE, MAE, and SSR.
"""
class CARTDecisionTreeRegressor:
    """
    Initialises the decision tree with hyperparameters.
        
    Parameters:
        max_depth: Maximum depth of the tree (None for unlimited)
        min_samples_split: Minimum samples required to split a node
        min_samples_leaf: Minimum samples required in a leaf node
        criterion: Splitting criterion - 'mse', 'mae', or 'ssr'
            - mse: Mean squared error (variance reduction)
            - mae: Mean absolute error
            - ssr: Sum of squared residuals
        max_features: Number of features to consider at each split
            - None: all features
            - 'sqrt': sqrt(n_features)
            - 'log2': log2(n_features)
            - int: exact number
            - float: fraction of features
    """
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, criterion='mse', max_features=None):
        
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion.lower()
        self.max_features = max_features
        self.root = None
        self.n_features = None
        self.n_features_split = None  # Actual number of features to consider at each split
        self.feature_importances_ = None
        
        # Validates criterion
        valid_criteria = ['mse', 'mae', 'ssr']
        if self.criterion not in valid_criteria:
            raise ValueError(f"criterion must be one of {valid_criteria}, got '{criterion}'")
    
    """
    Builds the decision tree from training data.
        
    Parameters:
        X: Training features
        y: Training target
    """    
    def fit(self, X, y):
        
        # Converts to numpy arrays if needed
        X = np.array(X)
        y = np.array(y)
        
        # Gets number of features and initialises feature importances
        self.n_features = X.shape[1]
        self.n_features_split = self._get_n_features_split()
        self.feature_importances_ = np.zeros(self.n_features)
        
        # Builds tree recursively
        self.root = self._build_tree(X, y, depth=0)
        
        # Normalises feature importances
        if self.feature_importances_.sum() > 0:
            self.feature_importances_ /= self.feature_importances_.sum()
        
        return self
    
    """
    Determines the number of features to consider at each split.
        
    Returns:
        Integer number of features to consider
    """
    def _get_n_features_split(self):
        if self.max_features is None:
            return self.n_features
        elif self.max_features == 'sqrt':
            return max(1, int(np.sqrt(self.n_features)))
        elif self.max_features == 'log2':
            return max(1, int(np.log2(self.n_features)))
        elif isinstance(self.max_features, int):
            return min(self.max_features, self.n_features)
        elif isinstance(self.max_features, float):
            return max(1, int(self.max_features * self.n_features))
        else:
            return self.n_features
    
    """
    Recursively builds the decision tree.
        
    Parameters:
        X: Feature matrix for current node
        y: Target values for current node
        depth: Current depth in the tree
        
    Returns:
        Node object
    """
    def _build_tree(self, X, y, depth):
        
        n_samples = len(y)
        
        # Checks stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           n_samples < 2 * self.min_samples_leaf: # Ensures enough samples for both children
            # Creates leaf node with appropriate value based on criterion
            return Node(value=self._calculate_leaf_value(y))
        
        # Finds the best split
        best_feature, best_threshold, best_gain = self._find_best_split(X, y)
        
        # If no valid split found, creates leaf node
        if best_feature is None:
            return Node(value=self._calculate_leaf_value(y))
        
        # Updates feature importance
        self.feature_importances_[best_feature] += best_gain * n_samples
        
        # Splits the data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        # Builds child nodes recursively
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return Node(
            feature=best_feature,
            threshold=best_threshold,
            left=left_child,
            right=right_child
        )
    
    """
    Finds the best feature and threshold to split on.
        
    Parameters:
        X: Feature matrix
        y: Target values
        
    Returns:
        best_feature: Index of best feature
        best_threshold: Threshold value for split
        best_gain: Reduction in impurity from the split
    """
    def _find_best_split(self, X, y):
        
        n_samples = len(y)
        parent_impurity = self._calculate_impurity(y)
        
        best_gain = 0
        best_feature = None
        best_threshold = None
        
        # Selects random subset of features to consider at this split
        if self.n_features_split < self.n_features:
            feature_indices = np.random.choice(self.n_features, self.n_features_split, replace=False)
        else:
            feature_indices = np.arange(self.n_features)
        
        # Iterates over selected features only
        for feature_idx in feature_indices:
            # Gets unique values for this feature
            feature_values = X[:, feature_idx]
            thresholds = np.unique(feature_values)
            
            # Tries each threshold
            for threshold in thresholds:
                # Splits the data
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)
                
                # Checks minimum samples in leaf constraint
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue
                
                # Calculates weighted impurity of children
                left_impurity = self._calculate_impurity(y[left_mask])
                right_impurity = self._calculate_impurity(y[right_mask])
                
                weighted_child_impurity = (n_left * left_impurity + n_right * right_impurity) / n_samples
                
                # Calculates gain (reduction in impurity)
                gain = parent_impurity - weighted_child_impurity
                
                # Updates best split if this is better
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    """
    Calculates impurity for a set of values based on the chosen criterion.
        
    Parameters:
        y: Target values
        
    Returns:
        Impurity value (MSE, MAE, or SSR)
    """
    def _calculate_impurity(self, y):
        
        if len(y) == 0:
            return 0
        
        if self.criterion == 'mse':
            # Mean squared error: average of squared deviations from mean
            mean = np.mean(y)
            return np.mean((y - mean) ** 2)
        
        elif self.criterion == 'mae':
            # Mean absolute error: average of absolute deviations from median
            # Uses median as it minimises MAE
            median = np.median(y)
            return np.mean(np.abs(y - median))
        
        elif self.criterion == 'ssr':
            # Sum of squared residuals: total squared deviation from mean
            mean = np.mean(y)
            return np.sum((y - mean) ** 2)
    
    """
    Calculates the leaf prediction value based on criterion.
    For MSE/SSR uses mean, for MAE uses median.
        
    Parameters:
        y: Target values
        
    Returns:
        Predicted value for leaf node
    """
    def _calculate_leaf_value(self, y):
        
        if self.criterion == 'mae':
            # Median minimises MAE
            return np.median(y)
        else:
            # Mean minimises MSE and SSR
            return np.mean(y)
    
    """
    Predicts target values for input samples.
        
    Parameters:
        X: Feature matrix
        
    Returns:
        Array of predictions
    """
    def predict(self, X):
        
        X = np.array(X)
        predictions = [self._traverse_tree(x, self.root) for x in X]
        return np.array(predictions)
    
    """
    Traverses the tree to make a prediction for a single sample.
        
    Parameters:
        x: Single sample features
        node: Current node in traversal
        
    Returns:
        Predicted value
    """
    def _traverse_tree(self, x, node):
        
        # If leaf node, returns the value
        if node.is_leaf():
            return node.value
        
        # Decides which child to visit
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)
    
    """Returns the maximum depth of the tree."""
    def get_depth(self):
        return self._get_depth(self.root)
    
    """Recursively calculates tree depth."""
    def _get_depth(self, node):
        if node is None or node.is_leaf():
            return 0
        # 1 + max depth of children because current node adds one level
        return 1 + max(self._get_depth(node.left), self._get_depth(node.right))
    
    """Returns the number of leaf nodes.
       Not redundant with _count_leaves since that is recursive helper.
    """
    def get_n_leaves(self):
        return self._count_leaves(self.root)
    
    """Recursively counts leaf nodes."""
    def _count_leaves(self, node):
        if node is None:
            return 0
        if node.is_leaf():
            return 1
        return self._count_leaves(node.left) + self._count_leaves(node.right)

"""
Creates a CART decision tree regressor with specified hyperparameters.
    
Parameters:
    max_depth: Maximum depth of the tree (None for unlimited)
    min_samples_split: Minimum samples required to split a node
    min_samples_leaf: Minimum samples required in a leaf node
    criterion: Splitting criterion - 'mse', 'mae', or 'ssr'
    
Returns:
    CARTDecisionTreeRegressor instance
"""
def create_decision_tree(max_depth=None, min_samples_split=2, min_samples_leaf=1, criterion='mse'):
    
    model = CARTDecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion
    )
    return model

"""
Trains the decision tree on the training data.
    
Parameters:
    model: CARTDecisionTreeRegressor instance
    X_train: Training features
    y_train: Training target
    
Returns:
    Trained model
"""
def train_decision_tree(model, X_train, y_train):
    
    model.fit(X_train, y_train)
    print(f"Trained decision tree with {model.get_n_leaves()} leaves and depth {model.get_depth()}")
    return model

"""
Evaluates the model and returns performance metrics.
    
Parameters:
    model: Trained CARTDecisionTreeRegressor
    X: Features
    y: True target values
    dataset_name: Name for display purposes
    
Returns:
    Dictionary with evaluation metrics
"""
def evaluate_model(model, X, y, dataset_name="dataset"):
    
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
Retrieves and displays feature importances from the trained tree.
    
Parameters:
    model: Trained CARTDecisionTreeRegressor
    feature_names: List of feature names
    
Returns:
    Dictionary mapping feature names to importance scores
"""
def get_feature_importance(model, feature_names):
    
    importances = model.feature_importances_
    
    # Creates sorted list of (feature, importance) tuples
    feature_importance = list(zip(feature_names, importances))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    print("\nFeature importances:")
    for feature, importance in feature_importance:
        if importance > 0:
            print(f"  {feature}: {importance:.4f}")
    
    return dict(feature_importance)
