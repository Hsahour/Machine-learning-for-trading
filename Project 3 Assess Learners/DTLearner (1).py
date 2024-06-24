import numpy as np

class DTLearner(object):
    def __init__(self, leaf_size=1, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.tree = None

    def author(self):
        return "hsahour3"

    def add_evidence(self, data_x, data_y):
        data = np.column_stack((data_x, data_y))
        self.tree = self.build_tree(data)

    def build_tree(self, data):
        if data.shape[0] <= self.leaf_size:
            return np.array([[-1, np.mean(data[:, -1]), np.nan, np.nan]])
        if np.all(data[:, -1] == data[0, -1]):
            return np.array([[-1, data[0, -1], np.nan, np.nan]])

        best_feature_index = self.best_feature_to_split(data)
        split_val = np.median(data[:, best_feature_index])

        if np.all(data[:, best_feature_index] == data[0, best_feature_index]):
            return np.array([[-1, np.mean(data[:, -1]), np.nan, np.nan]])

        left_data = data[data[:, best_feature_index] <= split_val]
        right_data = data[data[:, best_feature_index] > split_val]

        if len(left_data) == 0 or len(right_data) == 0:
            return np.array([[-1, np.mean(data[:, -1]), np.nan, np.nan]])

        left_tree = self.build_tree(left_data)
        right_tree = self.build_tree(right_data)

        if left_tree.ndim == 1:
            left_tree = np.array([left_tree])
        if right_tree.ndim == 1:
            right_tree = np.array([right_tree])

        root = np.array([[best_feature_index, split_val, 1, left_tree.shape[0] + 1]])
        return np.vstack((root, left_tree, right_tree))

    def best_feature_to_split(self, data):
        best_corr = 0
        best_feature_index = -1
        for i in range(data.shape[1] - 1):
            if np.std(data[:, i]) == 0:  
                continue
            corr = np.corrcoef(data[:, i], data[:, -1])[0, 1]
            if np.isnan(corr):  
                continue
            if abs(corr) > abs(best_corr):
                best_corr = corr
                best_feature_index = i
        return best_feature_index


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
    print("module DTlearn")
