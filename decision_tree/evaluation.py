import numpy as np
import decision_tree_learning as dtl

# ---------- K-fold cross-validation functions ----------

def kfold_spliting_stratified(data, k, seed, unique_labels, labels):
# Stratified k-fold splitting:
    # Ensures each fold has approximately the same class label proportions
    # as the overall dataset.
    # (This method gave the best and most stable results, so it is the one mainly used).

    #1. Initialize the random generator and storage lists
    rng = np.random.default_rng(seed)
    folds = []
    fold_indices = [[] for _ in range(k)] #stores class-balanced test indices for each fold
     
    #2. Loop over each unique class label: distribute its samples evenly across the k-folds
    for label in unique_labels:
        #2.1 Get indices of samples belonging to this label
        class_indices = np.where(labels == label)[0]
        #2.2 Shuffle class indices to randomize distribution
        rng.shuffle(class_indices)
        #2.3 Split this class's indices evenly across k folds 
            # (np.array_split distributes leftover samples across the first few folds)
        class_parts = np.array_split(class_indices, k)
        #2.4 Add each class split to the corresponding fold
        for i in range(k):
            fold_indices[i].extend(class_parts[i])

    #3. Create (train, test) index pairs for each fold
    for i in range(k):
        #3.1 The current fold's indices are used as test data
        test_indices = np.array(fold_indices[i])
        #3.2 The remaining samples are used as training data
        train_indices = np.setdiff1d(np.arange(len(data)), test_indices)
        #3.3 Store both sets for this fold 
        folds.append((train_indices, test_indices))

    #4. Return the list of folds
    return folds

def kfold_spliting(data, k, seed,shuffle=False):
# Simple k-fold splitting (non-stratified):
     # Randomly divides the dataset into k folds. Optionally shuffles the data first.
     #  (This method was used in earlier experiments; however, since it produced
     #  less consistent results compared to the stratified version, we kept it
     #  here for completeness and reference as mentioned in the report).

     # 1. Determine the total number of samples and storage lists
     n_samples = len(data)
     indices = np.arange(n_samples)
     folds = [] 

     # 2. Optionally shuffle the indices for random fold assignment
     if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

     #3. Split indices into k roughly equal folds
     folds_indices = np.array_split(indices, k)

     #4. Create (train, test) index pairs for each fold
     for i in range(k):
        #4.1 The current fold's indices are used as test data
        test_indices = folds_indices[i]
        #4.2 The remaining samples are used as training data
        train_indices = np.setdiff1d(np.arange(len(data)), test_indices)
        #4.3 Store both sets for this fold 
        folds.append((train_indices, test_indices))
    #5. Return the list of folds
     return folds

# ---------- Evaluations metrics functions ----------

def confusion_matrix(y_true, y_pred,n_classes):
    #1. Create an empty n_classes x n_classes confusion matrix
    cm = np.zeros((n_classes, n_classes), dtype=int)

    #2. Count predictions vs true labels
    min_label = y_true.min() # Handle cases where labels don’t start at 0 or 1    
    for t, p in zip(y_true.astype(int), y_pred.astype(int)):
        cm[t-min_label, p-min_label] += 1
    return cm

def accuracy(cm):
    # Accuracy = Correct Prediction (TP + TN) / All Prediction (TP + TN + FP + FN)
    return float(np.trace(cm)/cm.sum()) if cm.sum() > 0 else 0.0

def precision(cm):
    # Precision = TP / (TP + FP) per class
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) != 0)
    return precision

def recall(cm):
    # Recall = TP / (TP + FN) per class
    tp = np.diag(cm).astype(float)
    fn = cm.sum(axis=1) - tp
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) != 0)
    return recall

def f_one(cm):
    # F1-Score = 2 * (precision * recall) / (precision + recall)
    pre = precision(cm)
    rec = recall(cm)
    f1 = np.divide(
        2 * pre * rec,
        pre + rec,
        out=np.zeros_like(pre),
        where=(pre + rec)!= 0 )
    return f1

# ---------- Main CV runner ----------

def run_cv_metrics(dataset_path, k_folds, seed, max_depth=None, log_depths=True):
    #1. Load data
    data = np.loadtxt(dataset_path)
    X, y = data[:, :-1], data[:, -1].astype(int)

    #2. Get label info
    unique_labels = np.unique(y)
    n_classes = len(unique_labels)

    #3. Build stratified folds 
    folds = kfold_spliting_stratified(data, k_folds, seed, unique_labels, y)
    #folds = kfold_spliting(data, k_folds, seed,True) #Previous experiment

    #4. Storage for metrics all over folds
    cm_total = np.zeros((n_classes,n_classes), dtype=int)
    depths = []
    fold_conf_matrices = []

    #5. Loop over each fold
    for i, (tr_idx, te_idx) in enumerate(folds, 1):
        #5.1 Build training matrix
        train_matrix = np.c_[X[tr_idx], y[tr_idx]]
        #5.2 Train decision tree
        tree, depth = dtl.decision_tree_learning(train_matrix, max_depth=max_depth)
        #5.3 Record the depth of the tree for later analysis (Optional)
        if log_depths:
            print(f"[Fold {i}] depth: {depth}")
            depths.append(depth)
        #5.4 Predict on test fold
        y_pred = dtl.predict(tree, X[te_idx])
        #5.5 Compute confusion matrix and accuracy for this fold for later analysis (Optional)
        cm_fold = confusion_matrix(y[te_idx], y_pred,n_classes)
        fold_acc = accuracy(cm_fold)
        print(f"[Fold {i}] Accuracy: {fold_acc:.4f}")
        print(f"[Fold {i}] Confusion matrix:\n{cm_fold}\n")
        #5.6 Accumulate the confusion matrices across all folds
        cm_total += cm_fold
        #5.7 Store each fold's confusion matrix (Optional)
        fold_conf_matrices.append(cm_fold)

    #6. Average the confusion matrices
    cm_avg = cm_total / k_folds

    #7. Compute metrics from averaged confusion matrix 
    acc = accuracy(cm_avg)
    prec = precision(cm_avg)
    rec = recall(cm_avg)
    f1 =  f_one(cm_avg)
    macro_p, macro_r, macro_f1 = float(prec.mean()), float(rec.mean()), float(f1.mean())
    #7.1 Bundle results
    out = {
        "confusion_matrix": cm_avg,
        "accuracy": acc,
        "per_class_precision": prec,
        "per_class_recall": rec,
        "per_class_f1": f1,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1
    }
    #7.2  Append  the depth of each decision tree to the results for later analysis (Optional)
    if log_depths: 
        out["depths"] = depths
        out["average_depth"] = np.sum(depths)/ k_folds
    return out


def run_cv_metrics_pruned(dataset_path, k_folds, seed, max_depth=None, log_depths=True):
    #1. Load data
    data = np.loadtxt(dataset_path)
    X, y = data[:, :-1], data[:, -1].astype(int)

    #2. Get label info
    unique_labels = np.unique(y)
    n_classes = len(unique_labels)

    #3. Build stratified OUTER folds
    outer_folds = kfold_spliting_stratified(data, k_folds, seed, unique_labels, y)

    #4. Storage for metrics all over folds
    cm_total_outer = np.zeros((n_classes, n_classes), dtype=int)
    outer_depths = []

    #5. Loop over each OUTER fold
    for outer_i, (outer_train_idx, outer_test_idx) in enumerate(outer_folds, 1):
        #5.1 Split data into outer train pool and outer test set
        outer_train_matrix = np.c_[X[outer_train_idx], y[outer_train_idx]]
        outer_test_X = X[outer_test_idx]
        outer_test_y = y[outer_test_idx]

        #5.2 Perform INNER CV on the outer training data only.
        #Split the outer training set into (k_folds - 1) inner folds:
        #one fold for validation and the rest for training.
        X_outer_train = outer_train_matrix[:, :-1]
        y_outer_train = outer_train_matrix[:, -1].astype(int)
        inner_labels = np.unique(y_outer_train)
        inner_folds = kfold_spliting_stratified(
            outer_train_matrix,
            #5.2.1 Inner CV uses one fold for validation
            k_folds - 1,         
            #5.2.2 Different seed per outer fold (keeps it reproducible)
            seed*10+outer_i,          
            inner_labels,
            y_outer_train
        )

        #5.3 Collect results from each inner split, all evaluated on THE SAME outer_test set
        cm_total_inner_for_this_outer = np.zeros((n_classes, n_classes), dtype=int)
        inner_depths = []

        #5.4 Loop over each INNER fold
        for inner_j, (inner_train_idx, inner_val_idx) in enumerate(inner_folds, 1):
            #5.4.1 Build inner train and validation sets (matrices with features+label)
            inner_train_matrix = outer_train_matrix[inner_train_idx]
            inner_val_matrix   = outer_train_matrix[inner_val_idx]

            #5.4.2 Train full decision tree inner_train_matrix
            tree, depth = dtl.decision_tree_learning(inner_train_matrix,max_depth=max_depth)

            #5.4.3 Prune using the validation split (inner_val_matrix)
            tree_pruned, depth_pruned = dtl.tree_pruning(tree, inner_val_matrix)
            
            #5.4.4 Record the depth of the pruned tree for later analysis (Optional)
            if log_depths:
                inner_depths.append(depth_pruned)

            #5.4.5 Evaluate pruned tree on the OUTER TEST set
            y_pred_outer = dtl.predict(tree_pruned, outer_test_X)

            #5.4.6 Compute confusion matrix
            cm_inner = confusion_matrix(outer_test_y, y_pred_outer, n_classes)

            #5.4.7 Accumulate the confusion matrices across all INNER folds
            cm_total_inner_for_this_outer += cm_inner

        #5.5 Average INNER results for this OUTER test fold
        cm_avg_for_outer = cm_total_inner_for_this_outer / (k_folds - 1)

        #5.6 Accumulate into grand total (across outer folds)
        cm_total_outer += cm_avg_for_outer.astype(int)

        #5.7 Record the average depth of each OUTER decision tree later analysis(Optional)
        if log_depths and len(inner_depths) > 0:
            avg_inner_depth = np.mean(inner_depths)
            print(f"[Outer Fold {outer_i}] Avg depth: {np.round(avg_inner_depth, 3)}")
            outer_depths.append(avg_inner_depth)

        #5.8 Compute confusion matrix and accuracy for this OUTER fold for later analysis(Optional)
        fold_acc = accuracy(cm_avg_for_outer)
        print(f"[Outer Fold {outer_i}] Avg pruned accuracy on outer test set: {fold_acc:.4f}")
        print(f"[Outer Fold {outer_i}] Avg confusion matrix:\n{np.round(cm_avg_for_outer, 2)}\n")

    #6. Average across OUTER folds as well
    cm_final = cm_total_outer / k_folds

    #7. Compute metrics from final averaged confusion matrix
    acc_final = accuracy(cm_final)
    prec_final = precision(cm_final)
    rec_final  = recall(cm_final)
    f1_final   = f_one(cm_final)
    macro_p = float(prec_final.mean())
    macro_r = float(rec_final.mean())
    macro_f1 = float(f1_final.mean())
    #7.1 Bundle results
    results = {
        "confusion_matrix": cm_final,
        "accuracy": acc_final,
        "per_class_precision": prec_final,
        "per_class_recall": rec_final,
        "per_class_f1": f1_final,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1
    }
    #7.2 Append the average depth of each OUTER decision to the results tree later analysis(Optional)
    if log_depths and len(outer_depths) > 0:
        results["avg_pruned_depth_per_outer_fold"] = outer_depths
        results["average_depth"] = np.sum(outer_depths)/ k_folds
    return results