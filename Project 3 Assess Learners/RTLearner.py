import numpy as np
import random


class RTLearner(object):
    def __init__(self, leaf_size=1, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose
        random.seed(903941641)
        self.tree = None

    def author(self):
        return 'hsahour3'

    def add_evidence(self, data_x, data_y):
        data = np.column_stack((data_x, data_y))
        self.tree = self.build_tree(data)

    def build_tree(self, data):
        if data.shape[0] <= self.leaf_size or np.all(data[:, -1] == data[0, -1]):
            return np.array([[-1, np.mean(data[:, -1]), np.nan, np.nan]])

        else:
            features = list(range(data.shape[1] - 1))
            np.random.shuffle(features)
            best_feature = features[0]
            split_val = np.median(data[:, best_feature])
            if np.all(data[:, best_feature] == data[0, best_feature]):
                return np.array([[-1, np.mean(data[:, -1]), np.nan, np.nan]])

            left_data = data[data[:, best_feature] <= split_val]
            right_data = data[data[:, best_feature] > split_val]

            if left_data.size == 0 or right_data.size == 0:
                return np.array([[-1, np.mean(data[:, -1]), np.nan, np.nan]])

            left_tree = self.build_tree(left_data)
            right_tree = self.build_tree(right_data)

            if left_tree.ndim == 1:
                left_tree = np.array([left_tree])
            if right_tree.ndim == 1:
                right_tree = np.array([right_tree])

            root = np.array([[best_feature, split_val, 1, left_tree.shape[0] + 1]])
            return np.vstack((root, left_tree, right_tree))

    def query(self, points):
        predictions = np.array([self.query_single_point(point) for point in points])
        return predictions

    def query_single_point(self, point):
        node = 0
        while self.tree[node, 0] != -1:
            feature_index = int(self.tree[node, 0])
            if point[feature_index] <= self.tree[node, 1]:
                node += int(self.tree[node, 2])
            else:
                node += int(self.tree[node, 3])
        return self.tree[node, 1]
    
if __name__ == "__main__":  
    print("RTlearner module")
