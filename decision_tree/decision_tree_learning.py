import numpy as np
from decision_tree_node import DecisionTreeNode

# ---------- Core splitting and information gain functions ----------

def Entropy(dataset):
    # take labels from the last column
    labels = dataset[:, -1]
    # count unique labels and make probabilities
    values, counts = np.unique(labels, return_counts=True)
    
    probabilities = counts / counts.sum()
    # standard entropy with a tiny epsilon to avoid log(0)
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-9))
    
    return entropy

def remainder(dataset_left, dataset_right):
    # weighted entropy after the split (left/right)
    n_left, n_right = len(dataset_left), len(dataset_right)
    n = n_left + n_right
    if n == 0:
        return 0.0
    return (n_left / n) * Entropy(dataset_left) + (n_right / n) * Entropy(dataset_right)


def Gain(dataset_all, dataset_left, dataset_right):
    # information gain = parent entropy - weighted children entropy
    return Entropy(dataset_all) - remainder(dataset_left, dataset_right)


def information_gain(dataset_all, attribute_index, split_value):
    # compute gain for a given (feature, threshold)
    left_set, right_set = split_dataset(dataset_all, attribute_index, split_value)
    # if split is degenerate (all to one side), mark as bad
    if len(left_set) == 0 or len(right_set) == 0:
        return -np.inf
    return Gain(dataset_all, left_set, right_set)

def split_dataset(dataset, attribute_index, split_value):
    # binary split: <= goes left, > goes right
    right_condition = dataset[:, attribute_index] > split_value
    left_condition = dataset[:, attribute_index] <= split_value
    
    right_set = dataset[right_condition]
    left_set = dataset[left_condition] 
    
    return left_set, right_set

def find_split(train_dataset):
    # search over all features and all midpoints to find best gain
    attribute_number = train_dataset.shape[1] - 1

    best_gain = -np.inf
    best_feature = None
    best_threshold = None

    for i in range(attribute_number):
        # sort by this feature to scan consecutive values
        arguments = np.argsort(train_dataset[:, i], kind="mergesort")
        col_sorted = train_dataset[arguments, i]

        curr_max = -np.inf
        curr_thr = None

        # try midpoints between distinct consecutive values
        for j in range(len(arguments) - 1):
            val = col_sorted[j]
            next_val = col_sorted[j + 1]
            if val == next_val:
                continue

            mid_point = (val + next_val) / 2.0

            i_g = information_gain(train_dataset, i, mid_point)

            # track per-feature best
            if i_g > curr_max:
                curr_max = i_g
                curr_thr = mid_point

            # track global best
            if i_g > best_gain:
                best_gain = i_g
                best_feature = i
                best_threshold = mid_point
                
    # if nothing useful found, return none
    if best_feature is None or not np.isfinite(best_gain) or best_gain <= 0.0:
        return None, None, None, None
    
    # build the actual left/right branches for the best split
    l_branch, r_branch = split_dataset(train_dataset, best_feature, best_threshold)
    
    # safety: ensure split didn't collapse to empty side
    if len(l_branch) == 0 or len(r_branch) == 0:
        return None, None, None, None
    
    
    return l_branch, r_branch, best_feature, float(best_threshold)

# ---------- Prediction helper functions ----------

def predict_one(node, x):
    current = node

    while current is not None and not current.is_leaf():
        # If this node doesn't have a valid split anymore (e.g. after pruning),
        # stop here and classify using this node's stored majority class.
        if current.feature is None or current.threshold is None:
            break

        # Decide which branch to take
        if x[current.feature] <= current.threshold:
            # If left branch is missing (can happen after pruning), stop here
            if current.left is None:
                break
            current = current.left
        else:
            # If right branch is missing, stop here
            if current.right is None:
                break
            current = current.right

    # At this point:
    # - either current.is_leaf() was True
    # - or we broke early and we'll use its best guess
    # We return a label, guaranteed NOT None
    if current.room_label is not None:
        return current.room_label
    if current.no_classes is not None:
        return current.no_classes

    # This should basically never happen since we sets no_classes everywhere
    return 0

def predict(node, X):
    return np.array([predict_one(node, xi) for xi in X])


# ---------- Decision tree learner function ----------
    
def decision_tree_learning(training_dataset, depth=0, max_depth=None, fallback_label=None):
    n = len(training_dataset)

    # Empty node: return a leaf predicting the parent's majority class
    if n == 0:
        return DecisionTreeNode(room_label=fallback_label, depth=depth, no_classes=fallback_label), depth

    # labels at this node
    y = training_dataset[:, -1]

    # majority label at this node (break ties by min to be deterministic)
    vals, cnts = np.unique(y, return_counts=True)
    pred = vals[np.argwhere(cnts == cnts.max()).flatten()].min() if len(vals) > 0 else fallback_label

    # stop if pure or hit max depth → make a leaf with majority label
    if np.all(y == y[0]) or (max_depth is not None and depth >= max_depth):
        return DecisionTreeNode(room_label=pred, depth=depth, no_classes=pred), depth
    
    # try to split
    l_branch, r_branch, feature, threshold = find_split(training_dataset)

    # if no valid split, fallback to leaf with majority
    if feature is None:
        return DecisionTreeNode(room_label=pred, depth=depth, no_classes=pred), depth

    # recurse on children as per the psuedocode
    left_node,  l_depth = decision_tree_learning(l_branch, depth + 1, max_depth, pred)
    right_node, r_depth = decision_tree_learning(r_branch, depth + 1, max_depth, pred)

    # make internal node with info instead of dictionary
    node = DecisionTreeNode(
        feature=feature,
        threshold=threshold,
        left=left_node,
        right=right_node,
        depth=depth,
        no_classes=pred 
    )
    return node, max(l_depth, r_depth)

# ---------- Decision tree pruning function ----------

def tree_pruning(node, validation_dataset):
  
    # return node if it's a leaf 
    if node.is_leaf():
        return node, node.depth
    
    # split the validation data using the tree's condition 
    left_node, right_node  = split_dataset(validation_dataset, node.feature, node.threshold)
    
    # recursive pruning (inn case of missing children)
    if node.left is not None:
        node.left, l_depth = tree_pruning(node.left, left_node)
    else:
        l_depth = node.depth
    if node.right is not None:
        node.right, r_depth = tree_pruning(node.right, right_node)
    else:
        r_depth = node.depth

    # if no validation samples reach this node, prefer pruning (simpler model)
    if len(validation_dataset) == 0:
        return DecisionTreeNode(room_label=node.no_classes, depth=node.depth, no_classes=node.no_classes), node.depth
    
    # otherwise , evaluate: keep subtree vs prune to leaf
    y_true = validation_dataset[:, -1]
    X_val = validation_dataset[:, :-1]
   
    # 1. accuracy if we KEEP the subtree
    y_pred_subtree = predict(node, X_val) 
    acc_subtree = accuracy(y_true, y_pred_subtree)

    # 2. accuracy after turning the node into a leaf, The no_classes is the majority class from traning
    y_pred_leaf = np.full(len(y_true), node.no_classes)
    acc_leaf = accuracy(y_true, y_pred_leaf)

    # compare between 1,2
    if acc_leaf >= acc_subtree:
        # if leaf is better (pruned) update the node into leaf, and update the max depth
        return DecisionTreeNode(room_label=node.no_classes, depth=node.depth, no_classes=node.no_classes), node.depth
    else:
        # if subtree is better (original) keep it
        depth = max(l_depth, r_depth)
        return node, depth

def accuracy(y_true, y_pred):
    return float((y_true == y_pred).mean())