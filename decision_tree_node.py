# ---------- Decision tree node ----------
class DecisionTreeNode:
    def __init__(self, feature=None, threshold=None, room_label=None, left=None, right=None, leaf=None, depth=0, no_classes = None):
        self.feature = feature
        self.threshold = threshold
        self.room_label = room_label
        self.left = left
        self.right = right
        if room_label is not None:
            self.leaf = True
        else:
            self.leaf = None
        self.depth = depth
        self.no_classes = no_classes
    
    def is_leaf(self):
        return self.leaf is not None
