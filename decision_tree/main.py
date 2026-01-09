import sys
import numpy as np
import time
import evaluation as eval

# ---------- main CW1 runner ----------
if __name__ == "__main__":
    
    if len(sys.argv) <= 1:
      #1. Define dataset file paths
      clean_path = "./wifi_db/clean_dataset.txt"   
      noisy_path = "./wifi_db/noisy_dataset.txt"   

      #BEFORE PRUNING

      #2. Evaluate the decision tree with a 10-fold cross-validation on the CLEAN DATASET
      print("\nCLEAN | 10-fold CV before pruning")
      #2.1 Record start time to measure runtime for CLEAN dataset (Optional)
      start_time = time.time()
      #2.2 Run 10-fold cross-validation on CLEAN dataset
      #     - k_folds=10: given number
      #     - seed=42: ensures reproducibility
      #     - max_depth=None: no pre-pruning (grow full tree)
      #     - log_depths=True: prints tree depth per fold
      results_clean = eval.run_cv_metrics(dataset_path=clean_path, k_folds=10, seed=42, max_depth=None, log_depths=True)
      #2.3 Record end time and compute elapsed time for later analysis (Optional)
      end_time = time.time()
      clean_time = end_time - start_time
      #2.4 Print overall confusion matrix and performance metrics (Macro is Optinal)
      print("Averaged Confusion matrix:\n", results_clean["confusion_matrix"])
      print("Accuracy:", round(results_clean["accuracy"], 4))
      print("Averaged Depth:", np.round(results_clean["average_depth"],3))
      print("Per-class Precision:", np.round(results_clean["per_class_precision"], 3))
      print("Per-class Recall   :", np.round(results_clean["per_class_recall"], 3))
      print("Per-class F1       :", np.round(results_clean["per_class_f1"], 3))
      print("Macro P/R/F1       :", round(results_clean["macro_precision"],3),
                                    round(results_clean["macro_recall"],3),
                                    round(results_clean["macro_f1"],3))
      #2.5 Display total time taken for CLEAN dataset
      print(f"Time taken (CLEAN): {clean_time:.2f} seconds")

      #3. Evaluate the decision tree with a 10-fold cross-validation on the NOISY DATASET
      print("\nNOISY | 10-fold CV before pruning")
      start_time = time.time()
      results_noisy = eval.run_cv_metrics(dataset_path=noisy_path, k_folds=10,seed=42, max_depth=None, log_depths=True)
      end_time = time.time()
      noisy_time = end_time - start_time
      print("Averaged Confusion matrix:\n",  results_noisy["confusion_matrix"])
      print("Accuracy:", round(results_noisy["accuracy"], 4))
      print("Averaged Depth:", np.round(results_noisy["average_depth"],3))
      print("Per-class Precision:", np.round(results_noisy["per_class_precision"], 3))
      print("Per-class Recall   :", np.round(results_noisy["per_class_recall"], 3))
      print("Per-class F1       :", np.round(results_noisy["per_class_f1"], 3))
      print("Macro P/R/F1       :", round(results_noisy["macro_precision"],3),
                                    round(results_noisy["macro_recall"],3),
                                    round(results_noisy["macro_f1"],3))
      print(f"Time taken (NOISY): {noisy_time:.2f} seconds")

      #AFTER PRUNING

      #4. Evaluate the pruned decision tree using nested 10-fold cross-validation on the CLEAN DATASET.
      print("\nCLEAN | Nested 10-fold CV after pruning")
      #4.1 Record start time to measure runtime for CLEAN dataset after pruning (Optional)
      start_time = time.time()
      #4.2 Run nested 10-fold cross-validation with pruning on CLEAN dataset
      #    - Outer loop: evaluates generalization (test fold)
      #    - Inner loop: trains + prunes using a validation fold
      #    - The confusion matrices are averaged across all inner/outer folds
      results_clean_pruned = eval.run_cv_metrics_pruned( dataset_path=clean_path,k_folds=10,seed=42,max_depth=None,log_depths=True)
      #4.3 Record end time and compute elapsed time
      end_time = time.time()
      clean_time_pruned = end_time - start_time
      #4.4 Print overall confusion matrix and performance metrics for the PRUNED model (Macro is Optinal)
      print("Averaged Confusion matrix (pruned):\n", results_clean_pruned["confusion_matrix"])
      print("Accuracy (pruned):", round(results_clean_pruned["accuracy"], 4))
      print("Averaged Depth (pruned):", np.round(results_clean_pruned["average_depth"],3))
      print("Per-class Precision (pruned):", np.round(results_clean_pruned["per_class_precision"], 3))
      print("Per-class Recall    (pruned):", np.round(results_clean_pruned["per_class_recall"], 3))
      print("Per-class F1        (pruned):", np.round(results_clean_pruned["per_class_f1"], 3))
      print("Macro P/R/F1 (pruned):",
            round(results_clean_pruned["macro_precision"], 3),
            round(results_clean_pruned["macro_recall"],    3),
            round(results_clean_pruned["macro_f1"],        3))
      #4.5 Display total time taken for CLEAN dataset (after pruning)
      print(f"Time taken (CLEAN, pruned): {clean_time_pruned:.2f} seconds")

      #5. Evaluate the pruned decision tree using nested 10-fold cross-validation on the NOISY DATASET
      print("\nNOISY | Nested 10-fold CV after pruning")
      start_time = time.time()
      results_noisy_pruned = eval.run_cv_metrics_pruned( dataset_path=noisy_path, k_folds=10,seed=42,max_depth=None,log_depths=True)
      end_time = time.time()
      noisy_time_pruned = end_time - start_time
      print("Averaged Confusion matrix (pruned):\n", results_noisy_pruned["confusion_matrix"])
      print("Accuracy (pruned):", round(results_noisy_pruned["accuracy"], 4))
      print("Averaged Depth (pruned):", np.round(results_noisy_pruned["average_depth"],3))
      print("Per-class Precision (pruned):", np.round(results_noisy_pruned["per_class_precision"], 3))
      print("Per-class Recall    (pruned):", np.round(results_noisy_pruned["per_class_recall"], 3))
      print("Per-class F1        (pruned):", np.round(results_noisy_pruned["per_class_f1"], 3))
      print("Macro P/R/F1 (pruned):",
            round(results_noisy_pruned["macro_precision"], 3),
            round(results_noisy_pruned["macro_recall"],    3),
            round(results_noisy_pruned["macro_f1"],        3))
      print(f"Time taken (NOISY, pruned): {noisy_time_pruned:.2f} seconds")

    else:
      # This part is used to test the secret dataset
      print(sys.argv[0],sys.argv[1]) #sys.argv[1] is the file plath
      path = sys.argv[1]

      #BEFORE PRUNING

      print(f"\n{path} | 10-fold CV before pruning")
      start_time = time.time()
      results = eval.run_cv_metrics(dataset_path=path, k_folds=10, seed=42, max_depth=None, log_depths=True)
      end_time = time.time()
      tim_taken = end_time - start_time
      print("Averaged Confusion matrix:\n", results["confusion_matrix"])
      print("Accuracy:", round(results["accuracy"], 4))
      print("Averaged Depth:", np.round(results["average_depth"],3))
      print("Per-class Precision:", np.round(results["per_class_precision"], 3))
      print("Per-class Recall   :", np.round(results["per_class_recall"], 3))
      print("Per-class F1       :", np.round(results["per_class_f1"], 3))
      print("Macro P/R/F1       :", round(results["macro_precision"],3),
                                    round(results["macro_recall"],3),
                                    round(results["macro_f1"],3))
      print(f"Time taken ({path}): {tim_taken:.2f} seconds")

      #AFTER PRUNING

      print(f"\n{path} | Nested 10-fold CV after pruning")
      results_pruned = eval.run_cv_metrics_pruned(dataset_path=path,k_folds=10,seed=42,max_depth=None,log_depths=True)
      end_time = time.time()
      time_taken_pruned = end_time - start_time
      print("Averaged Confusion matrix (pruned):\n", results_pruned["confusion_matrix"])
      print("Accuracy (pruned):", round(results_pruned["accuracy"], 4))
      print("Averaged Depth (pruned):", np.round(results_pruned["average_depth"],3))
      print("Per-class Precision (pruned):", np.round(results_pruned["per_class_precision"], 3))
      print("Per-class Recall    (pruned):", np.round(results_pruned["per_class_recall"], 3))
      print("Per-class F1        (pruned):", np.round(results_pruned["per_class_f1"], 3))
      print("Macro P/R/F1 (pruned):",
            round(results_pruned["macro_precision"], 3),
            round(results_pruned["macro_recall"],    3),
            round(results_pruned["macro_f1"],        3))
      print(f"Time taken ({path}, pruned): {time_taken_pruned:.2f} seconds")