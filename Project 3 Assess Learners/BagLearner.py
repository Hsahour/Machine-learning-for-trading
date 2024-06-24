import numpy as np
                           
class BagLearner(object):
    def __init__(self, learner, kwargs={}, bags=20, boost=False, verbose=False):
        self.learner = learner
        self.kwargs = kwargs
        self.bags = bags
        self.boost = boost
        self.verbose = verbose
        self.learners = [learner(**kwargs) for _ in range(bags)]

    def author(self):
        return "hsahour3"  

    def add_evidence(self, Xtrain, Ytrain):
        for learner in self.learners:
            indices = np.random.choice(Xtrain.shape[0], size=Xtrain.shape[0])
            bootstrap_x = Xtrain[indices]
            bootstrap_y = Ytrain[indices]
            learner.add_evidence(bootstrap_x, bootstrap_y)

    def query(self, Xtest):
        predictions = np.array([learner.query(Xtest) for learner in self.learners])
        return predictions.mean(axis=0)

if __name__ == "__main__":  
    print("BagLearner module")
