class Node:
    def __init__(self, feature=None, value=None):
        self.feature = feature
        self.value = value
        self.children = []

    def __repr__(self):
        feature = self.feature if self.feature is not None else "None"
        value = self.value if self.value is not None else "None"
        return f'Node({feature}={value})'
    
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
        return self._repr_recursive(self.root, 0)
        
    def _repr_recursive(self, node, spaces):
        result = f"{node}"
        new_spaces = spaces + len(str(node))
        for i, child in enumerate(node.children):
            if i == 0:
                result += f" -> {self._repr_recursive(child, new_spaces)}"
            else:
                result += f"\n{' ' * new_spaces} -> {self._repr_recursive(child, new_spaces)}"
        return result

    def get_depth(self):
        return self.root.get_depth()