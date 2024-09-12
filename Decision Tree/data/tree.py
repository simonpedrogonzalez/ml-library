class Node:
    def __init__(self, feature=None, value=None):
        self.feature = feature
        self.value = value
        self.children = []

    def __repr__(self):
        return f'Node({self.feature}={self.value})'
    
    def add_child(self, node):
        self.children.append(node)
    
    def is_leaf(self):
        return len(self.children) == 0
    
    def get_depth(self):
        return self._get_depth(self)
    
    def _get_depth(self, node):
        if node.is_leaf():
            return 0
        return 1 + max([self._get_depth(child) for child in node.children])

class Tree:
    def __init__(self, root):
        self.root = root
    
    def __repr__(self):
        return f'Tree({self.root})'

    def get_depth(self):
        return self.root.get_depth()