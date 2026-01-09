# Intro to Machine Learning – CW1 Decision Trees

## Setup

### 1. Activate the Virtual Environment

Run the following command on the lab machines to activate the provided virtual environment:

```bash
source /vol/lab/ml/intro2ml/bin/activate
```

### 2. Install Dependencies

Once the environment is active, install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Code

### Run on Both Datasets (Default)

If you run `main.py` without any arguments, the program will train and evaluate on both the clean and noisy datasets:

```bash
python main.py
```

### Run on a Specific Dataset

To test the program on a specific dataset, provide the dataset file path as an argument:

```bash
python main.py <filepath>
```

**Example:**

```bash
python main.py ./wifi_db/clean_dataset.txt
```

---

## Bonus: Decision Tree Visualization

To visualize the decision tree trained on the entire clean dataset, run:

```bash
python tree_visualizer.py
```

This will display a graphical representation of the trained decision tree. Please full screen the window to see the same result as the image in the report.

---

## File Overview

| File | Description |
|------|--------------|
| `main.py` | Main script for training and evaluating the decision tree. |
| `decision_tree_learning.py` | Contains the decision_tree_learning() function along side pruning logic |
| `decision_tree_node.py` | Defines the structure and properties of a decision tree node. |
| `evaluation.py` | Contains evaluation metrics and performance testing. |
| `tree_visualizer.py` | Visualizing funciton for decision tree. |
| `requirements.txt` | Lists all Python dependencies required to run the project. |

---