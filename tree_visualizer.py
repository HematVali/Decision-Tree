import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from decision_tree_learning import decision_tree_learning

# ---------- Layout utilities function ----------
def assign_positions_layers(root):
    # bfs over the tree, group nodes by depth (layers), then assign x indexes per layer
    pos = {}
    if root is None:
        return pos

    #initialize queue 
    queue = deque([(root, 0)])
    layers = {}

    #classic bfs
    while queue:
        node, depth = queue.popleft()
        #store layer by layer to make it easier to format
        layers.setdefault(depth, []).append(node)

        left = node.left
        right = node.right

        if left is not None:
            queue.append((left, depth + 1))
        if right is not None:
            queue.append((right, depth + 1))

    for depth, nodes in layers.items():
        n = len(nodes)

        #base case for the root node
        if n == 1:
            x_positions = [0.0]
        else:
            #stats for spacing will store -1.5, 0 1.5 ect
            spacing = 1.0 
            total_width = (n - 1) * spacing
            start_x = -total_width / 2.0
            x_positions = [start_x + i * spacing for i in range(n)]

        #store the data
        for node, x in zip(nodes, x_positions):
            pos[id(node)] = (x, depth)

    return pos

# ---------- Drawing and formatting helpers functions ----------
def _draw_node(ax, x, y, text, box_w=3.0, box_h=1.0, leaf=False):
    bbox_props = dict(
        boxstyle="round,pad=0.3",
        facecolor="#119B25" if leaf else "#F5F5F5",
        edgecolor="#333333",
        linewidth=1.0
    )
    #text inside the box format
    ax.text(
        x, y, text,
        ha="center", va="center",
        fontsize=9,
        bbox=bbox_props
    )

def format_node_text(node, feature_names=None):
    #label nodes as leaf
    if node.is_leaf():
        label = node.room_label
        return f"Leaf\nRoom = {int(label)}"

    #retrieve the node stats
    feature = node.feature
    threshold = node.threshold

    if feature_names is None:
        feature_name = f"A{feature}"
    else:
        feature_name = feature_names[feature]

    return f"{feature_name} ≤ {threshold:.2f}"

# ---------- Visualize tree function ----------
def visualize_tree(root, feature_names=None, figsize=(26, 14), x_pad=6.5, y_step=3.0, node_size=(4.5, 1.0), show_axis=False):
    #retrieve the position of root node
    pos = assign_positions_layers(root)

    scaled_pos = {}

    #retreve positions of nodes
    for key, (x_index, depth) in pos.items():
        x_scaled = x_index * x_pad
        y_scaled = -depth * y_step
        scaled_pos[key] = (x_scaled, y_scaled)

    pos = scaled_pos

    fig, ax = plt.subplots(figsize=figsize)

    #get rid of axis
    if not show_axis:
        ax.axis("off")

    #set view limits with a little padding
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    pad = 1.0
    ax.set_xlim(min(xs) - 2 * pad, max(xs) + 2 * pad)
    ax.set_ylim(min(ys) - (node_size[1] + pad), max(ys) + (node_size[1] + pad))

    # draw edges first so nodes sit on top
    def _draw_edges(node):
        if node is None or node.is_leaf():
            return
        x0, y0 = pos[id(node)]
        left = node.left
        right = node.right
        for child in (left, right):
            if child is not None:
                x1, y1 = pos[id(child)]
                # connect bottom of parent box to top of child box
                ax.plot([x0, x1],
                        [y0 - node_size[1] / 2.0, y1 + node_size[1] / 2.0],
                        linewidth=1.0, color="#888888")
                _draw_edges(child)

    _draw_edges(root)

    # draw node boxes and labels
    def _draw_nodes(node):
        if node is None:
            return
        x, y = pos[id(node)]
        text = format_node_text(node, feature_names)
        _draw_node(ax, x, y, text,
                   box_w=node_size[0],
                   box_h=node_size[1],
                   leaf=node.is_leaf())
        left = node.left
        right = node.right
        _draw_nodes(left)
        _draw_nodes(right)

    _draw_nodes(root)

    plt.tight_layout()
    plt.show()


# ---------- main Tree Visualizer runner ----------
if __name__ == "__main__":
    # load the clean dataset
    data = np.loadtxt('./wifi_db/clean_dataset.txt')

    # train a full tree
    tree, max_d = decision_tree_learning(data, max_depth=None)

    # simple feature names: X0, X1, ..., X(d-1)
    feat_names = [f"X{i}" for i in range(data.shape[1] - 1)]

    # draw the tree with slightly larger canvas and spacing
    visualize_tree(tree, feature_names=feat_names, figsize=(28, 16), node_size=(4.8, 1.1), x_pad=7.2, y_step=3.2, show_axis=False)